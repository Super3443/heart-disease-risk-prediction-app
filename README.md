# Predicting Heart Disease Presence

# 1. Overview

- This project develops a machine-learning model to predict whether heart disease may be present using patient assessment data.

- The final tuned model is deployed in a Streamlit decision-support application that displays a prediction and model-estimated probability. The application supports clinical review and does not replace professional medical diagnosis.

- The final tuned Gradient Boosting Classifier achieved `84.8%` testing accuracy, `92.2%` heart-disease recall, `82.5%` precision and an `87.0%` F1-score, with `8` false negatives and `20` false positives.

# 2. The Dataset

This project uses the `heart_disease_uci.xlsx` dataset from Kaggle.
Dataset link: [Heart Disease UCI](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data)

# 3. Progress Log

## 17-07-2026:

- Setup 
    - Imported the heart_disease_uci.xlsx dataset into the Jupyter Notebook file
    - Imported libraries
    - Added the Business Understanding section

## 18-07-2026:

- Data Visualisation Section
   - Completed Understanding of Target
        - Bar chart of the original heart disease target values
        - Bar chart of the converted binary target

    - Completed Understanding of Features
        - Bar plot of missing values by feature
        - Boxplots for continuous features
        - Count plots for categorical features

    - Completed Understanding relationship of variables
        - Correlation heatmap for numerical features
        - Pairplot for continuous features by target group
        - 100% stacked bar charts showing heart disease proportion by categorical feature

- Data Cleaning Section
    - Initial Data Cleaning
    - Split X and y
    - Performed Train-test split (80/20 ratio)
    - Handling missing values
    - Performed One-Hot-Encoding (OHE)
    - Feature Scaling

## 19-07-2026:
### First Iteration (Logistic Regression Model)

- Modelling Section
    - Selected `Logistic Regression` as the baseline classification model
    - Trained the model using the scaled training data

- Model Evaluation Section
    - Achieved `81.5%` testing accuracy
    - Achieved `85%` recall, `82%` precision and `84%` F1-score for heart disease
    - Created a confusion matrix
    - Identified `15` false negatives and `19` false positives
    - Compared training and testing accuracy with no clear signs of overfitting

- Completed the first model iteration

### Second Iteration (Model Comparisons)

- Reused the same prepared dataset and train-test split

- Modelling Section
    - Added 3 additional classification models
    - Decision Tree Classifier
    - Random Forest Classifier
    - Gradient Boosting Classifier

- Model Evaluation Section
    - Compared all models using the same dataset and train-test split from the first iteration
    - Recorded the accuracy, precision, recall, F1-score and false negatives

    - Gradient Boosting Classifier achieved the strongest results
        - `84.2% `testing accuracy
        - `90.2%` recall
        - `82.9%` precision
        - `86.4%` F1-score
        - `10` false negatives

- Conclusion
    - Decision Tree and Random Forest showed signs of overfitting
    - Selected Gradient Boosting Classifier as the leading model


## 20-07-2026:
### Third Iteration (Missing-Value Column Feature Engineering)

- Added missing-value features for ca, slope and thal

- Retrained and compared all four classification models

- Gradient Boosting remained the strongest model
    - `84.8%` testing accuracy
    - `90.2%` recall
    - `86.8%` F1-score
    - `10` false negatives

### Fourth Iteration (Hyperparameter Tuning)

- Used RandomizedSearchCV to tune Logistic Regression, Decision Tree, Random Forest and Gradient Boosting

- Used model-specific combinations of four hyperparameters for each algorithm

- Prioritised heart-disease recall using five-fold stratified cross-validation

- Compared each tuned model with its untuned Iteration 3 baseline

- Selected the best Gradient Boosting settings:
    - n_estimators: `100`
    - learning_rate: `0.05`
    - max_depth: `3`
    - max_features: `1.0`

- Tuned Gradient Boosting Classifier results:
    - `84.8%` testing accuracy
    - `92.2%` recall
    - `82.5%` precision
    - `87.0%` F1-score
    - `8` false negatives

- Conclusion
    - Gradient Boosting provided the strongest balance of recall, precision, F1-score and testing accuracy
    - Reduced false negatives from `10` to `8`
    - Selected the tuned Gradient Boosting Classifier as the final model for the Streamlit application

### Deployment Stage

- Tested final model on new unseen data to observe the results

- Saved model and processing information named "heart_disease_model_gbt_rs.pkl" to the "model" folder

- Selected the saved final model for use in the Streamlit decision-support application

## 23-07-2026:

### Streamlit Website Development

- Created a Streamlit decision-support application

- Added a patient assessment form using the 13 original dataset features

- Applied the same missing-value handling and one-hot encoding used during model training

- Loaded the final tuned Gradient Boosting Classifier from heart_disease_model_gbt_rs.pkl

- Displayed the predicted class and model-estimated probability

- Added input validation and a medical-use disclaimer

- Added a transparent heart icon and responsive light and dark theme styling

## 28-7-2026

### Improvements

- Improved code comments, rationales and model evaluation explanations.

- Added feature-importance and cumulative-importance graphs for all four tuned models.

- Added a `sex` representation comparison and documented the limitation caused by fewer female records.

## 29-7-2026

### Final Model Review

## 29-07-2026

### Final Notebook Review

- Increased the Decision Tree, Random Forest and Gradient Boosting tuning searches to `50` sampled combinations.

- Evaluated all `24` available Logistic Regression combinations.

- Added a table showing the three strongest tuning configurations for each algorithm.

- Added a four-iteration modelling roadmap.

- Improved the feature-retention rationale and documented why all `21` encoded features were retained.

- Added a separate assumptions and limitations section.

- Added a final project summary containing the selected model and testing results.

- Checked that the observations, tables and graphs use matching values.

- Added concise comments throughout the data analysis, modelling, tuning, unseen-data and deployment code.

