import ast


def summarize(llm, max_retries=10):
    # max_retries is used for rerun the llm call if the output is not in the correct format (sometimes llm can output invalid json etc)
    def summarizer(state):
        """
        Reviews the manuscript and data, both in ASCII format, and provides a summary, a descriptive key
        for the data fields, and notes on both.

        Args:
            manuscript_path: path for the manuscript text file
            data_path: path for the data text file

        Returns:
            returns a dictionary with keys 'manuscript_summary', 'column_key', and 'notes'. 'column key' is a nested dictionary.

        """

        # state: AutoFeaturizer class instance
        manuscript_path = state.manuscript_path
        data_path = state.data_path

        # Read the manuscript file
        try:
            with open(manuscript_path, 'r', encoding='utf-8') as f:
                manuscript_text = f.read()
        except Exception as e:
            print(f"Error reading manuscript file: {e}")
            return

        # Read the data file
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data_lines = f.readlines()[:5]  # Read first 5 lines
                data_text = ''.join(data_lines)
        except Exception as e:
            print(f"Error reading data file: {e}")
            return

        # ---------------- SYSTEM PROMPT ----------------
        system_message = (
            "System: You are tasked with understanding and summarizing a scientific text with particular attention to the use of the data for development of machine learning models.\n\n"
            "Problem summary: Analyze the manuscript and data files provided in the user message\n"
            "Output format (STRICT): Provide output in the following nested dictionary format\n"
            "\n==== Output format ====\n"
            "{\n"
            " 'manuscript_summary':'<generated summary>',\n"
            " 'column_key': {\n"
            "                '<column 1 name>': '<column 1 physical interpretation and notes>',\n"
            "                '<column 2 name>': '<column 2 physical interpretation and notes>',\n"
            "                ...\n"
            "                '<column M name>': '<column M physical interpretation and notes>',\n"
            "               },\n"
            " 'notes': '<any notes or context that you think is important to relay>'\n\n"
            "}\n"
        )

        # ---------------- USER PROMPT ----------------
        user_msg = (
            "\n==== Manuscript text ====\n"
            f"{manuscript_text}\n"
            "==== Data ====\n"
            f"{data_text}\n"
            "\n\nInstructions:\n"
            "No extra instructions necessary\n"
        )

        prompt = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_msg}
            ]

        # Retry loop:
        def is_valid_result(result):

            resd = ast.literal_eval(result)
            try:
                check1 = 'manuscript_summary' in resd.keys()
                check2 = 'column_key' in resd.keys()
                check3 = 'notes' in resd.keys()
                check4 = len(resd['column_key'].keys()) > 1
                check = check1 and check2 and check3 and check4

            except Exception as e:
                print(f'json improperly formated: {resd}')
                return False

            return check
        
        for _ in range(max_retries):
            raw = llm(prompt)
            resd = ast.literal_eval(raw)
            if is_valid_result(raw):
                state.literature_review = resd['manuscript_summary']  # update the state with the result
                state.features_description = resd['column_key']      # update the state with the result
                return
        # If we exhaust all retries, we can return the an error or raise an exception
        raise RuntimeError(f"Failed after {max_retries} retries. Last output: {raw}")

    return summarizer
