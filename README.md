# AutoFeatSci: Automated Feature Engineering for Materials Science  

## 📖 Overview  

Materials science research often involves two key ingredients:  
1. **Knowledge from the literature** (e.g., physics behind alloying, defect formation, thermodynamics).  
2. **Structured experimental or simulation datasets** (e.g., composition, microstructure, processing conditions).  

Traditionally, connecting these two requires significant **manual effort**: scientists read papers, design features by hand, and then test them in ML models.  

**AutoFeatSci** automates this process with a **multi-agent LLM pipeline** that integrates domain reasoning with machine learning (ML).  

Given a materials science paper 📄 and an associated dataset 📊, plus a **target property** to predict (e.g., hardness, hot cracking susceptibility, conductivity), AutoFeatSci:  

- Leverages a **literature-grounded reasoning module** powered by LLMs to **synthesize and distill insights** from materials science publications, extracting governing mechanisms, descriptors, and domain hypotheses.  
- Employs a **data-aware interpretation module** that uses LLMs to **contextualize dataset features**, mapping raw variables to their underlying physical significance and linking them to knowledge derived from the literature.  
- Generates **novel feature hypotheses** by **cross-referencing scientific insights with dataset semantics**, enabling construction of physically meaningful descriptors that go beyond naïve feature engineering.  
- Validates proposed features through **systematic predictive modeling**, assessing their statistical and physical relevance with performance metrics (e.g., RMSE, R²) and feature importance analysis.  
- Operates in an **iterative closed-loop framework**, where evaluation feedback continuously informs new feature hypotheses, driving convergence toward an optimized feature space aligned with the prediction target.  


---

## 🎯 Motivation  

This project aims to solve a bottleneck in **materials informatics**:  
- Scientists are often constrained by their ability to handcraft meaningful features.  
- Literature contains valuable knowledge (relationships, trends, known descriptors) that are rarely encoded directly into datasets.  
- Feature engineering is time-consuming and error-prone.  

By automating this loop, AutoFeatSci helps:  
- Accelerate **hypothesis testing** by rapidly trying out physically meaningful features.  
- Provide **explainable mappings** between features and literature insights.  
- Close the gap between raw experimental/simulation data and predictive ML pipelines.  

---

## ⚙️ The Pipeline  

AutoFeatSci is organized as a **multi-agent network**, where each agent has a specialized role.  

### 1. **Paper Analyzer  and Raw Feature Description Agent** (`summarize`)
- Reads in raw manuscript text and data files
- Prepares summary of manuscript to be used in downstream tasks, as well as a succint description of each feature present in the original data, highlighting its physical significance to the task at hand

### 2. **Feature Proposal Agent** (`proposal`)  
- **Generates Physically Meaningful Features:**  Analyzes existing features, literature summaries, target definitions, and past model performance to propose new features that are physically interpretable and relevant to the prediction task.
- **Specifies Feature Derivation:**  Provides clear instructions on how to compute each proposed feature from the original dataset, ensuring reproducibility and integration into the feature generation pipeline.


### 3. **Feature Generation Agent** (`execution`)
- Translates proposed feature hypotheses into concrete dataset transformations, ensuring that each candidate feature is materialized as a new column in the DataFrame.
- Executes transformations reliably using standard numerical operations, while preserving the integrity of the original dataset.

### 4. **Evaluation Module**  
- Continuously evaluates the effectiveness of the current featurization by training predictive models on the augmented dataset.  
- Measures model performance using metrics such as RMSE, R², and feature importance to assess how well the engineered features capture the target property.  
- Summarizes results into feedback reports, which are passed back to the **Feature Proposal Agent** for refinement.  
- Based on this feedback, the proposal agent can accept the current features, reject underperforming ones, or propose new combinations for the next iteration.  


---

## 🚀 What This Project Brings  

- **LLM-driven AutoML**: not just blind feature generation, but guided by literature reading and reasoning.  
- **Closed-loop optimization**: features are iteratively refined based on predictive performance.  
- **Interpretable outputs**: every feature proposal is documented with its physical significance and provenance from literature.  
- **Scalable approach**: designed to be applied across different material systems and targets.  


---

## 🧭 Future Directions  

- Support additional ML models (XGBoost, GPs, Neural Nets).  
- Incorporate **uncertainty quantification** into feedback.  
- Extend paper analyzer to extract **explicit equations/relationships** from literature.  
- Apply to benchmark datasets (e.g., Materials Project, AFLOWLIB).  

---

## 🤝 Contributions  

This is an early-stage project. Contributions are welcome!  
- Extend the agent modules.  
- Add test datasets + example workflows.  
- Improve robustness of feature code generation.  


