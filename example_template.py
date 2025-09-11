def create_llm_condense_repair_agent(llm, max_retries=10):
    """
    Condense/repair agent for the ImageToy reconstruction problem:
    - Parameter vector length = n_vars (σ_k per dataset).
    - Objective vector length = n_objs (RMS residuals per dataset).
    - Returns a list of dicts: [{'values': [..n_vars..], 'rationale': '...'}, ...] with length == pool size.
    """

    def condense_repair_agent_node(state):
        parent_pool = state["parent_pool"]                 # (N, n_vars)
        parent_objectives = state["parent_objectives"]     # (N, n_objs)
        n_vars = parent_pool.shape[1]
        n_objs = parent_objectives.shape[1]
        n_keep = state.get("n_keep", 10)
        accept_threshold = state.get("accept_threshold", 0.2)
        max_objective_factor = state.get("max_objective_factor", 2.0)

        # Bounds string
        bounds = state.get("bounds", None)
        if bounds is not None:
            lower, upper = bounds
            bounds_str = f"[{float(np.min(lower))}, {float(np.max(upper))}]"
        else:
            bounds_str = "[0, 1]"

        # Preselect top-k by total error
        obj_sum = parent_objectives.sum(axis=1)
        best_indices = np.argsort(obj_sum)[:n_keep]
        selected_pool = parent_pool[best_indices]
        selected_objectives = parent_objectives[best_indices]

        # Summaries
        param_param_corr = state["summary"]["param_param_corr"]
        param_obj_corr = state["summary"]["param_obj_corr"]
        pca_loadings = state["summary"]["pca_loadings"]
        pca_explained_variance = state["summary"]["pca_explained_variance"]
        param_mean = state["summary"]["param_mean"]
        param_std = state["summary"]["param_std"]
        param_min = state["summary"]["param_min"]
        param_max = state["summary"]["param_max"]
        obj_mean = state["summary"]["obj_mean"]
        obj_std = state["summary"]["obj_std"]
        obj_min = state["summary"]["obj_min"]
        obj_max = state["summary"]["obj_max"]

        n_generated = parent_pool.shape[0] - n_keep

        # Identify bad sets
        def find_bad_params(parent_pool, parent_objectives, accept_threshold, max_objective_factor):
            max_allowed = accept_threshold * max_objective_factor
            bad_idx = np.where((parent_objectives > max_allowed).any(axis=1))[0]
            bad_info = []
            for idx in bad_idx:
                bad_dims = np.where(parent_objectives[idx] > max_allowed)[0].tolist()
                bad_info.append(
                    f"- Row {idx}: high RMS at objectives {bad_dims}; "
                    f"params={np.array2string(parent_pool[idx], precision=3, separator=', ')}, "
                    f"objs={np.array2string(parent_objectives[idx], precision=3, separator=', ')}"
                )
            return bad_info

        bad_info = find_bad_params(parent_pool, parent_objectives, accept_threshold, max_objective_factor)
        bad_summary = "\n".join(bad_info) if bad_info else "None found."

        # Pareto summary
        pareto_summary = (
            "Note: This summary provides an overview of the entire Pareto front, including key statistics for both parameters and objectives.\n"
            "\n(Each element in the arrays below corresponds to one input/output dimension; e.g. the first value relates to the statistic value for parameter/objective 0, the second to parameter/objective 1, etc.):\n"
            f"Entire Pareto front mean: {arr2str(param_mean)}\n"
            f"Entire Pareto front std: {arr2str(param_std)}\n"
            f"Entire Pareto front min: {arr2str(param_min)}\n"
            f"Entire Pareto front max: {arr2str(param_max)}\n"
            f"Pareto front objective mean: {arr2str(obj_mean)}\n"
            f"Pareto front objective std: {arr2str(obj_std)}\n"
            f"Pareto front objective min: {arr2str(obj_min)}\n"
            f"Pareto front objective max: {arr2str(obj_max)}\n"
            "Loadings: " + arr2str(pca_loadings) + "\n"
            "Explained variance: " + arr2str(pca_explained_variance) + "\n"
        )


        # ---------------- SYSTEM PROMPT ----------------
        system_message = (
            "System: You are an optimization agent tuning hyperparameters for a multi-dataset image reconstruction problem.\n\n"
            "Problem summary:\n"
            f"- Each candidate parameter vector has length {n_vars}: per-dataset scale parameters σ_k.\n"
            f"- Each reconstruction yields an objective vector of length {n_objs}: RMS residuals e_k for each dataset (lower is better).\n\n"
            "How parameters drive objectives:\n"
            "- The parameters σ_k act as scaling factors in the minimization process.\n"
            "- Smaller σ_k → dataset k has more influence, which may reduce its error but risks overfitting its noise and hurting other datasets.\n"
            "- Larger σ_k → dataset k has less influence, which may prevent overfitting but can leave its error high.\n"
            "- Your job: find σ values that reduce all RMS objectives without collapsing into overfitting on one dataset or ignoring others.\n\n"
            "What you will be given, and how it can help:\n"
            f"1) Full parameter pool (N×{n_vars}) and objective values (N×{n_objs}):\n"
            "   • Shows which σ patterns are linked to better or worse objectives.\n"
            "2) Very bad sets (with high RMS):\n"
            "   • Avoid those σ patterns in new proposals.\n"
            f"3) Parameter–parameter correlation ({n_vars}×{n_vars}):\n"
            "   • Entry [i, j] is the correlation between σ_i and σ_j across the population.\n"
            "   • Positive correlation: when σ_i is high, σ_j also tends to be high (they move together).\n"
            "   • Negative correlation: when σ_i is high, σ_j tends to be low (they trade off).\n"
            "   • Near zero: σ_i and σ_j vary independently.\n"
            "   • This reveals which σ often move together; Use them during your proposal if needed.\n"
            f"4) Parameter–objective correlation ({n_vars}×{n_objs}):\n"
            "   • Entry [i, j] is the correlation between σ_i and objective j (RMS error of dataset j).\n"
            "   • Shows how each σ influences each objective.\n"
            "   • To improve a target objective k, do not only consider σ_k itself—look at other σ_j that are strongly correlated with objective k.\n"
            "   • For example: if objective k is high and σ_j has a strong positive correlation with objective k, decreasing σ_j may also reduce objective k "
            "(and vice versa for negative correlation).\n"
            f"5) PCA loadings + explained variance:\n"
            "   • Each principal component (PC) shows a dominant direction of variation in the parameter space.\n"
            "   • If one σ dimension has a very large loading in the first PC (which explains most variance), treat that parameter as the highest-priority lever—it has the strongest effect on objectives.\n"
            "   • Use early PCs (with high explained variance) to guide parameter moves in promising regions.\n"
            "   • Later PCs (low explained variance) are less important, but may capture subtle trade-offs.\n"
            f"6) Global pareto front statistics summary:\n"
            "   • The Pareto front identifies the best non-dominated solutions of the entire history.\n"
            "   • Use these sets as anchors for guidance.\n"
            "   • Look at their parameter statistics: this shows what σ regions are currently most successful.\n"
            "   • Look at their objective values: this shows the achievable trade-offs so far.\n"
            "   • When generating new sets, use these Pareto anchors as references.\n"
            f"7) Selected top {n_keep} sets:\n"
            "   • Keep these best sets as-is; generate the rest to improve objectives further.\n\n"
            "**Guidelines:**\n"
            "- Learn from the information provided (correlations, PCA, Pareto sets, the current parameter and objective pools) to adjust σ in ways likely to reduce global errors.\n"
            "- When generating new sets: propose σ vectors you believe will lower overall error, guided by patterns in you have observed so far.\n"
            "- Be careful not to collapse into extreme values (all σ → 0 or very large), as that leads to overfitting or underfitting.\n"
            f"- You must keep the proposed σ values within the allowed bounds {bounds_str}.\n\n"
            "Output format (STRICT):\n"
            f"- Return a valid Python list of {parent_pool.shape[0]} dicts.\n"
            f"- Each dict must have 'values' (a list of {n_vars} floats) and 'rationale' (short text).\n"
            "- The FIRST LINE of your reply must be ONLY that Python list—no extra text."
        )


        # ---------------- USER PROMPT ----------------
        user_msg = (
            "\n==== Indexing & Semantics ====\n"
            f"• Parameter indices: 0..{n_vars-1} (σ_k per dataset).\n"
            f"• Objective indices:  0..{n_objs-1} (RMS error per dataset; lower is better).\n\n"
            "==== Parent Pool (parameters) ====\n"
            + arr2str(parent_pool, decimals=3, max_rows=20)
            + "\n\n==== Parent Objectives (RMS errors) ====\n"
            + arr2str(parent_objectives, decimals=3, max_rows=20)
            + "\n\n==== Very-High-Error Sets ====\n"
            + bad_summary
            + "\n\n==== Global Parameter–Parameter Correlation ====\n"
            + f"(Matrix shape: {n_vars}×{n_vars}).\n"
            + "- Entry [i, j] is the correlation between σ_i and σ_j across the population.\n"
            + arr2str(param_param_corr)
            + "\n\n==== Global Parameter–Objective Correlation ====\n"
            + arr2str(param_obj_corr)
            + "\n\n==== Global Pareto Front Summary + Stats ====\n"
            + pareto_summary
            + "\n\n==== Selected Top to Keep/Repair ====\n"
            "Parameters:\n" + arr2str(selected_pool, decimals=3, max_rows=n_keep) +
            "\nObjectives (RMS):\n" + arr2str(selected_objectives, decimals=3, max_rows=n_keep) +
            "\n\nInstructions:\n"
            f"• Keep these {n_keep} sets and generate {n_generated} new parameter sets (total = pool size).\n"
            "• Focus on reducing global RMS errors.\n"
            "• Do not collapse into extreme σ values (all → 0 or very large).\n"
            "• Keep values within bounds. Avoid duplicates.\n"
            "• The FIRST line of your reply must be ONLY the Python list of dicts."
        )


        # Retry loop
        tries = 0
        sets = None
        report_raw = ""
        while sets is None and tries < max_retries:
            prompt = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_msg}
            ]
            result = llm(prompt)
            report_raw = result
            try:
                sets = ast.literal_eval(result.strip())
                valid = (
                    isinstance(sets, list)
                    and all(
                        isinstance(item, dict)
                        and 'values' in item
                        and 'rationale' in item
                        and isinstance(item['values'], (list, tuple, np.ndarray))
                        and len(item['values']) == n_vars
                        for item in sets
                    )
                    and len(sets) == parent_pool.shape[0]
                )
                if not valid:
                    sets = None
            except Exception:
                sets = None

            if sets is None:
                system_message += (
                    f"\nWARNING: Your previous output was NOT a valid Python list of {parent_pool.shape[0]} dicts "
                    f"with 'values' (length {n_vars}) and 'rationale'. "
                    "The first line must be ONLY that Python list. Try again."
                )
            tries += 1

        if sets is None:
            print("LLM agent output parse error (final attempt).")
            print("Raw LLM output:", report_raw)
            return {
                **state,
                "condensed_pool": np.empty((0, parent_pool.shape[1])),
                "rationales": [],
                "condense_report_raw": report_raw
            }

        arrs = [np.array(item['values'], dtype=float) for item in sets]
        rationales = [str(item['rationale']) for item in sets]
        return {
            **state,
            "condensed_pool": np.vstack(arrs),
            "rationales": rationales,
            "condense_report_raw": report_raw
        }

    return condense_repair_agent_node
