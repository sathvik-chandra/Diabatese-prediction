**Diabetes Prediction Model**

---

### Overview:
This repository contains code for building a machine learning model to predict the likelihood of diabetes based on various health-related features. The dataset used in this project consists of demographic information, medical history, and other relevant factors.

### Dataset:
The dataset used for training and testing the model is stored in a CSV file named `diabetes_prediction_dataset.csv`. It contains the following columns:

1. **gender**: Gender of the individual (categorical)
2. **age**: Age of the individual (numerical)
3. **hypertension**: Indicates if the individual has hypertension (binary: 0 or 1)
4. **heart_disease**: Indicates if the individual has heart disease (binary: 0 or 1)
5. **smoking_history**: Smoking history of the individual (categorical)
6. **bmi**: Body Mass Index (numerical)
7. **HbA1c_level**: Hemoglobin A1c level (numerical)
8. **blood_glucose_level**: Blood glucose level (numerical)
9. **diabetes**: Target variable indicating the presence of diabetes (binary: 0 or 1)

### Libraries Used:
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- Imbalanced-learn (for handling class imbalance)
  
### Steps:
1. **Data Understanding**: Loading and understanding the dataset, including data types, missing values, and unique values.
2. **Data Preprocessing**:
   - One-hot encoding of categorical variables ('gender' and 'smoking_history').
   - Handling duplicate values.
   - Correlation analysis between features and target variable.
   - Visualization of distributions and correlations.
3. **Machine Learning**:
   - Direct Random Forest approach.
   - Hyperparameter tuning using GridSearchCV.
   - Handling class imbalance using SMOTE.
   - Hyperparameter tuning with SMOTE.
4. **Model Evaluation**:
   - Evaluation of model performance using accuracy and classification reports.

 Model Performance:
- **Direct Random Forest**:
  - Accuracy: 97%
- **HyperTuned Random Forest**:
  - Accuracy: 97%
- **SMOTE + Random Forest**:
  - Accuracy: 96%
- **SMOTE + Random Forest + HyperTuned**:
  - Accuracy: 96%

Usage:
1. Ensure you have the required libraries installed .
2. Clone the repository.
3. Run the Jupyter notebook or Python script to train and evaluate the model.

 Future Improvements:
- Feature engineering to enhance model performance.
- Trying out different machine learning algorithms.
- Fine-tuning hyperparameters further for better accuracy.

---

This README provides an overview of the project, including dataset information, steps followed, model performance, and usage instructions. For more detailed insights, refer to the Jupyter notebook or Python script provided in the repository.
