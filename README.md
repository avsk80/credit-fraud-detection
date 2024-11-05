# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions through advanced data analysis and machine learning techniques. We address the challenges of high dimensionality and class imbalance using state-of-the-art tools in Python, achieving interpretable, robust, and high-performing models. Our approach combines data preparation, feature extraction, model evaluation, and interpretability with SHAP to make predictive insights actionable.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Exploration and Analysis (EDA)](#data-exploration-and-analysis-eda)
3. [Feature Engineering & Dimensionality Reduction](#feature-engineering--dimensionality-reduction)
4. [Modeling and Evaluation](#modeling-and-evaluation)
5. [Model Calibration](#model-calibration)
6. [Interpretability with SHAP and Partial Dependence](#interpretability-with-shap-and-partial-dependence)
7. [Results](#results)
8. [Conclusion](#conclusion)
9. [Key Insights](#key-insights)

---

### Project Overview

Credit card fraud detection is a high-stakes problem, where accuracy in identifying fraud can prevent substantial financial losses. This project addresses this problem through the following workflow:
1. Exploratory Data Analysis (EDA)
2. Feature Engineering and Extraction (including PCA)
3. Model Selection and Evaluation
4. Model Calibration for Probability Interpretation
5. Interpretability Analysis with SHAP

---

### Data Exploration and Analysis (EDA)

Our analysis begins with an extensive **Exploratory Data Analysis (EDA)** to uncover key insights:
- **Class Distribution**: We confirmed severe class imbalance, as fraud cases are far fewer than non-fraud cases, requiring specialized handling.
- **Feature Analysis**: Using visualizations and statistical methods, we examined categorical and continuous features. Custom functions (e.g., `plot_cat_feat_dist`) were used to understand feature distributions and relationships with the target class.
- **Correlations and Statistical Tests**: Chi-square and ANOVA tests were used to identify features with strong statistical relevance, ensuring that our feature set supports accurate model predictions.

### Feature Engineering & Dimensionality Reduction

In the **Feature Engineering** phase, we aimed to make the data both more informative and computationally efficient:
- **Scaling and Encoding**: All numerical features were normalized, and categorical features were encoded.
- **Principal Component Analysis (PCA)**: To reduce dimensionality, we applied PCA on selected features. This was particularly useful for capturing variance in the data while reducing computational overhead.
- **Memory Optimization**: A custom `reduce_mem_usage` function minimized memory consumption, allowing us to handle the data more efficiently.

### Modeling and Evaluation

For **Model Selection**, we explored a variety of machine learning algorithms, emphasizing interpretability and accuracy:
- **Models**: We experimented with logistic regression, `RandomForestClassifier`, `XGBClassifier`, and `LGBMClassifier`.
- **Hyperparameter Tuning**: Using `GridSearchCV`, we optimized parameters to enhance model performance.
- **Evaluation Metrics**: Accuracy, ROC-AUC, and precision-recall curves were computed using custom `compute_evaluation_metric` functions to ensure an in-depth understanding of model performance.
  
Key Results from Modeling:
- **Best Model**: The XGBoost model with tuned hyperparameters achieved the highest ROC-AUC, indicating strong discriminatory power between fraud and non-fraud cases.
- **Precision-Recall Analysis**: Precision and recall scores demonstrated our model’s ability to balance false positives and false negatives effectively.

### Model Calibration

To make our model predictions more interpretable, we performed **Calibration**:
- **CalibratedClassifierCV** was applied to transform model outputs into well-calibrated probabilities, improving the interpretability of predictions.
- **Calibration Curves**: Calibration plots confirmed that the predicted probabilities are closely aligned with actual fraud likelihood, making these predictions actionable for business decisions.

### Interpretability with SHAP and Partial Dependence

Interpreting model predictions is critical for stakeholder trust. To that end, we employed **SHAP** and **Partial Dependence Analysis**:
- **SHAP Analysis**: Used for global and local interpretability, SHAP values allowed us to understand feature importance on both an aggregate and individual prediction basis. Techniques included:
  - **Force Plots**: Visualized the impact of each feature on individual predictions, highlighting drivers of fraud likelihood.
  - **Partial Dependence and Individual Conditional Expectation (ICE) Plots**: Showed how model predictions change with respect to single features, providing insights into feature interactions and non-linear relationships.
  - **Tree Interpreter and Feature Importance Plots**: Offered a breakdown of contributions from each feature, crucial for transparency.

### Results
- In this dataset, there are only 3.5% of fraud cases. This can be intuitively thought as, if we build a naive classifier that always outputs non-fraud the accuracy would be 96.5%. This can be thought of as our baseline.
#### Key Results and Visuals
- **Model Performance**:
  - XGBoost achieved the best overall results with an ROC-AUC of ~0.93.
  - Calibration curves validated that predicted probabilities were well-aligned with actual fraud rates.
  
- **SHAP Findings**:
  - Top Predictive Features: Transaction amount, past fraud history, and time-related variables (based on feature importance).
  - **Interpretability**: Force and ICE plots illustrated why specific transactions were classified as fraud, explaining feature influence in intuitive ways.

#### Sample Results Summary Table:
| Model                | Accuracy | ROC-AUC | Weighted Precision | Weighted Recall |
|----------------------|----------|---------|--------------------|-----------------|
| Random Forest        | 97.3%    | 0.88    | 0.91               |          0.65   |
| **XGBoost (Best)**   | 97.9%    | **0.93**| **0.94**           |         **0.73**|
| LightGBM             | 97.7%    | 0.93    | 0.92               |          0.70   |
| LightGBM(oversampled)| 88.7%    | 0.92    | 0.60               |          0.85   |

### Conclusion

Through this project, we achieved a highly accurate and interpretable fraud detection model. This solution is well-suited for real-world deployment, with calibrated probabilities enhancing decision-making for fraud prevention teams. The interpretability provided by SHAP further supports model transparency, which is essential in high-stakes financial applications.

### Key Insights

1. **Memory Efficiency**: Reducing memory usage allowed us to work with large datasets efficiently, a critical aspect of handling high-dimensional fraud data.
2. **Imbalanced Data Handling**: The use of `RandomOverSampler` effectively addressed the imbalance, boosting model recall for minority class. However, the overall accuracy was impacted resulting in a significantly less accurate model than the baseline
3. **Model Calibration for Trustworthy Predictions**: Calibrated probabilities made the model’s outputs more actionable, particularly in a domain where the cost of errors is high.
4. **SHAP-Based Interpretability**: The use of SHAP not only validated the model's predictions but also provided granular insights into fraud patterns, improving trust among stakeholders.

---

### Future Enhancements

- **Explore Additional Features**: Incorporating external data sources, like transaction locations or user demographics, could improve model accuracy.
- **Deploy as API**: Packaging the model as a real-time API would make it accessible for broader operational use.
- **Advanced Interpretability Methods**: Additional methods like LIME could complement SHAP, offering even more diverse interpretability perspectives.
