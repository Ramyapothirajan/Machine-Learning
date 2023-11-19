'''CRISP-ML(Q)

a. Business & Data Understanding
    As internet penetration is increasing the usage of eletronic media as mode of effective communication is increasing. 
    So are the spamsters who master in spamming your mailbox with innovative emails, which are difficult to classify as spam.
    A few of these might also have virus and might trick you into loosing money via fraud black hat techniques. 
    Same logic applies for Telecom companies when it comes to SMS - Short Messaging Service.

    i. Business Objective -  Maximize Predictive Accuracy
    ii. Business Constraint -  Minimize Model Complexity

    Success Criteria:
    1. Business Success Criteria - Reduce worker churn by at least 12%.
    2. ML Success Criteria - Achieve an salary prediction of over 80%
    3. Economic Success Criteria - Reduced Employee Turnover & Cost Savings through Talent Retention
    
    Data Collection - Salary collection data is obtained where the details of the employees are given.
    Training Dataset has 30161 observations and 14 columns. Testing Dataset has 15060 observations and 14 columns.
    
'''
import pandas as pd
import os
from sklearn.naive_bayes import GaussianNB,  MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report, recall_score, precision_score
from imblearn.over_sampling import SMOTE
import joblib 
from sqlalchemy import create_engine

# Credentials to connect to Database
user = 'root'  # user name
pw = 'Mypassword_23'  # password
db = 'univ_db'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
conn = engine.connect()

sal_train = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/11. Naive Bayes/SalaryData_Train.csv")
sal_test = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/11. Naive Bayes/SalaryData_Test.csv")

sal_test['Salary'].value_counts()
sal_train['Salary'].value_counts()

# Split data into features (X) and target variable (y)
X_train = sal_train.drop('Salary', axis=1)
y_train = sal_train['Salary']
X_test = sal_test.drop('Salary', axis=1)
y_test = sal_test['Salary']

# Perform one-hot encoding for categorical columns
X_train_encoded = pd.get_dummies(X_train, columns=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native'])
X_test_encoded = pd.get_dummies(X_test, columns=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native'])

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state = 0)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train)

# Train a Multinomial Naive Bayes classifier
classifier_mb = MultinomialNB()
classifier_mb.fit(X_train_resampled, y_train_resampled)

# Evaluate on Test Data
X_test_resampled = X_test_encoded  # No need to resample the test data
test_pred_m = classifier_mb.predict(X_test_resampled)

# Accuracy on Test Data
accuracy_test_m = accuracy_score(y_test, test_pred_m)

# Confusion Matrix on Test Data
confusion_matrix_test = confusion_matrix(y_test, test_pred_m)

# Accuracy on Training Data
train_pred_m = classifier_mb.predict(X_train_resampled)
accuracy_train_m = accuracy_score(y_train_resampled, train_pred_m)

# Confusion Matrix on Training Data
confusion_matrix_train = confusion_matrix(y_train_resampled, train_pred_m)

# Model Tuning - Hyperparameter optimization
mnb_lap = GaussianNB()
mnb_lap.fit(X_train_resampled, y_train_resampled)

# Evaluation on Test Data after applying laplace
test_pred_lap = mnb_lap.predict(X_test_resampled)

# Accuracy on Test Data after laplace
accuracy_test_lap = accuracy_score(y_test, test_pred_lap)

# Metrics
sensitivity_test = recall_score(y_test, test_pred_lap, pos_label=" >50K")  # Calculate sensitivity for ">50K" class
specificity_test = recall_score(y_test, test_pred_lap, pos_label=" <=50K")  # Calculate specificity for "<=50K" class
precision_test = precision_score(y_test, test_pred_lap, pos_label=" >50K")  # Calculate precision for ">50K" class

# Print results
print("Accuracy on Test Data (Multinomial NB):", accuracy_test_m)
print("Confusion Matrix on Test Data (Multinomial NB):\n", confusion_matrix_test)
print("Accuracy on Training Data (Multinomial NB):", accuracy_train_m)
print("Confusion Matrix on Training Data (Multinomial NB):\n", confusion_matrix_train)

print("Accuracy on Test Data after applying laplace (Multinomial NB):", accuracy_test_lap)
print("Sensitivity on Test Data (Multinomial NB) for '>50K':", sensitivity_test)
print("Specificity on Test Data (Multinomial NB) for '<=50K':", specificity_test)
print("Precision on Test Data (Multinomial NB) for '>50K':", precision_test)

# Confusion Matrix - Heat Map
cm_test = confusion_matrix(y_test, test_pred_lap)
cmplot_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['<=50K', '>50K'])

# Plot Confusion Matrix for Test Data
cmplot_test.plot()
cmplot_test.ax_.set(title='Salary Prediction Confusion Matrix',
                    xlabel='Predicted Value', ylabel='Actual Value')

# Save the trained model to a file
model_filename = 'nb_salary_model.joblib'
joblib.dump(classifier_mb, model_filename)

# Load the model for predictions (in another script or session)
loaded_model = joblib.load(model_filename)

# Example of using the loaded model for prediction
# Assuming you have a new data sample in X_new
prediction = loaded_model.predict(X_new)