# Fraud Detection

## Objective of the Project
The goal of the project is building a model using Machine learning that detects financial fraud by identifying suspicious activities.

## Dataset Overview:

https://www.kaggle.com/datasets/kartik2112/fraud-detection?select=fraudTrain.csv
The dataset is divided into two parts: train and test. 
It contains information about financial transactions, with a column indicating whether the transaction was fraudulent (target variable).

key Variables:
index - Unique Identifier for each row
trans_date_trans_time - Transaction DateTime
cc_num - Credit Card Number of Customer
merchant - Merchant Name
category - Category of Merchant
amt - Amount of Transaction
first - First Name of Credit Card Holder
last - Last Name of Credit Card Holder
gender - Gender of Credit Card Holder
street - Street Address of Credit Card Holder
city - City of Credit Card Holder
state - State of Credit Card Holder
zip - Zip of Credit Card Holder
lat - Latitude Location of Credit Card Holder
long - Longitude Location of Credit Card Holder
city_pop - Credit Card Holder's City Population
job - Job of Credit Card Holder
dob - Date of Birth of Credit Card Holder
trans_num - Transaction Number
unix_time - UNIX Time of transaction
merch_lat - Latitude Location of Merchant
merch_long - Longitude Location of Merchant
is_fraud - Fraud Flag <--- Target Class

## Challenges & Solutions
Multicollinearity between features - Solution: Addressed multicollinearity by  combining features and eliminating unnecessary ones.
Feature Selection - Solution: Kept features like ZIP, latitude, and longitude in the model,enhancing the feature engineering process for better fraud detection.

## Methodology

### **ETL: Data Extraction & Cleaning **
- Created a class named DataLoader that load the training and test datasets from CSV files.
- Combined both datasets into one DataFrame (combined_train_test_data)
- Created a class named DataExplorer that performs initial exploration of the combined data, displaying:the shape and columns of the dataset, Data types of each column and basic information about the DataFrame and the first five rows for a quick overview.
- Created a class named DataLoader that renames the columns for lowercase, while dealing with missinh values and duplicates and droped unnacessary columns.

### ** Exploratory Data Analysis, Data Preprocessing & Visualizations **
- Created a class named EDA that provides an overview of the dataset, including descriptive statistics and data visualizations
- Statistical summary: Computes and returns a statistical summary for numerical columns, including measures of central tendency (mean, median) and dispersion (variance, standard deviation, interquartile range (IQR), and outlier counts).
- Data Distribution Visualizations: distribution of numerical features (excluding 'is_fraud') using histograms with KDE plots and  distribution of the target variable 'is_fraud' using a count plot to compare fraudulent and non-fraudulent transactions.
- Inferential Statitics: The Chi-square test was applied to each categorical feature against the target variable (is_fraud) to check the association with the target, categorical variables tranformed for numeric and added to the correlation matrix.


### ** Feature Engineering **
- Created new features to the dataset to help address multicollinearity between existing features (time-related features, transaction amount features, and calculates the distance to the merchant)
- Additionally, was generated features like time differences, small city indicators, and combined categories, which provide more diverse information, reducing the correlation between the original features and improving model performance.

### ** Modelling  **
Train-Test Split:Divided the data into 70% for training and 30% for testing.
Scaling:Applies MinMaxScaler to normalize the numeric features in the training and test sets, scaling values between 0 and 1.
SMOTE (Over-sampling Technique):Balanced the dataset by generating synthetic samples for the minority class (fraud) in the training data.

Models used:

Logistic Regression:performs well in predicting non-fraud cases with perfect precision but misses 31% of actual non-fraud cases (recall 0.69). However, it struggles with fraud detection, showing extremely low precision (0.01) and F1-score (0.02), making it unreliable for detecting fraud.

Random Forest:performs  well on non-fraud cases with near-perfect precision and recall (0.99). While fraud detection improves over Logistic Regression, with 79% recall and a higher F1-score (0.51), its precision (0.38) remains low, limiting its reliability for fraud detection

XGBoost (model choosed)
The model performs well in detecting non fraudulent transactions, with 100% precision and recall. 
However, it only detects 81% of fraudulent transactions, missing 19% and having a 14% false positive rate. 
The high accuracy alone is not a reliable metric for this model due to the imbalanced data between the classes, where non-fraudulent transactions dominate.


### ** Data Visualizations **
- Feature Importance Plot : Used to check the impact the model's decisions for refining and guiding feature engineering.
- Confusion Matrix plot: The confusion matrix was important for evaluating the model's performance, particularly in identifying false positives
- ROC CURVE plot: We concluded that, themodel is good at distinguishing between fraudulent and non-fraudulent transactions (performing with very few mistakes) and can correctly classify +90% of the cases based on its predictions.

### ** Next Steps **
Review the specific false positives and false negatives to understand the model’s mistakes and improve data preprocessing.

### ** Delivarables
Jupyter Notebook with Full Code in Python: Contains all steps from data preprocessing to model evaluation.
PowerPoint Presentation: A concise overview of the project, including methodology, results, and key findings, supported by visual plots.
Power BI Dashboard: An interactive visualization of key metrics and model performance, featuring insights from the table with real and predicted values to explore patterns, trends, and key insights.
