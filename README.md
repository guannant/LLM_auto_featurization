# AutoFeatSci: Automated Feature Engineering for Materials Science  

## 📖 Overview  

Materials science research often involves two key ingredients:  
1. **Domain knowledge from literature** (e.g., physics behind alloying, defect formation, thermodynamics).  
2. **Structured experimental or simulation datasets** (e.g., composition, microstructure, processing conditions).  

Traditionally, connecting these two requires significant **manual effort**: scientists read papers, design features by hand, and then test them in ML models.  

**AutoFeatSci** automates this process with a **multi-agent AI pipeline** that integrates natural language processing (NLP) and machine learning (ML).  

Given a materials science paper 📄 and an associated dataset 📊, plus a **target property** to predict (e.g., hardness, hot cracking susceptibility, conductivity), AutoFeatSci:  

- Analyzes and summarizes the relevant literature.  
- Uses the summary and the dataset to **propose new feature candidates**.  
- Automatically generates the Python code to construct those features.  
- Evaluates the new features using ML (currently Random Forest).  
- Iteratively improves the feature set based on feedback.  

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

### 1. **Paper Analyzer Agent**  
- Reads the provided materials science paper.  
- Produces a concise **summary of domain knowledge**: important relationships, governing mechanisms, relevant descriptors.  

### 2. **Feature Description Agent**  
- Reviews the dataset (columns, distributions, metadata).  
- Generates a **semantic description** for each feature, linking it to physical meaning.  

### 3. **Feature Proposal Agent** (`feat_proposal`)  
- Takes as input:  
  - Feature descriptions.  
  - Paper summarization.  
  - Target property.  
  - Reports from previous model runs.  
- Proposes **new feature candidates** by combining existing features or designing transformations (e.g., ratios, normalized quantities).  
- Outputs two dictionaries:  
  - Physical meaning of each new feature.  
  - How to obtain them from the original dataset.  

### 4. **Feature Generation Agent**  
- Implements the proposal.  
- Automatically generates and executes **Python code** (using pandas/numpy) to construct the new features directly in the DataFrame.  
- Handles invalid instructions gracefully (filling NaNs when necessary).  

### 5. **Machine Learning Agent**  
- Runs a **Random Forest** model using the newly constructed features.  
- Evaluates model performance (RMSE, R², etc.) and feature importance.  

### 6. **Feedback Loop**  
- Model feedback (performance + feature importance) is summarized.  
- Passed back to the **Feature Proposal Agent** to guide the next iteration.  
- Loop continues until a specified number of iterations is reached.  

---

## 🚀 What This Project Brings  

- **Domain-aware AutoML**: not just blind feature generation, but guided by materials science literature.  
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


