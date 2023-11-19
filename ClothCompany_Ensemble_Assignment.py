'''
#CRISP-ML(Q) Framework: 

Business Understanding:

Business Problem: A cloth manufacturing company is interested to know about the different attributes contributing to high sales. 

Business Objective: Maximize Sales
Business Constraint: Maximize Customer Satisfaction

Success Criteria:
Business: Effective Marketing
ML: Achieve an accuracy of more than 85%
Economic: Reach profit more than 15%

Data Understanding: 
400 observations & 11 columns

10 columns are inputs & 1 column is output

1.  Sales: Number of units sold.
2.  CompPrice: Manufacturer's price for the product.
3.  Income: Average income in the product's or store's location.
4.  Advertising: Amount spent on advertising for the product or store.
5.  Population: The population in the area where the product or store is located.
6.  Price: Retail price at which the product is sold to customers.
7.  ShelveLoc: Shelf location of the product, categorized as "Bad," "Good," or "Medium."
8.  Age: Age of the store or product.
9.  Education: Education level of the population in the area.
10. Urban: Indicates whether the location is in an urban area (Yes/No).
11. US: Indicates whether the product is made in the United States (Yes/No).

'''

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import joblib
import sweetviz
import numpy as np

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_engine.outliers import Winsorizer
import sklearn.metrics as skmet

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import cross_validate


df = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/14. Ensemble Models/ClothCompany_Data.csv")

df.columns

df.info()

# Checking for Null values
df.isnull().sum()

#AutoEDA
report = sweetviz.analyze([df, "data"])
report.show_html('Report.html')

# Target variable categories
df['Sales'].value_counts()

# Data split into Input and Output
X = df.iloc[:, 1:11] # Predictors 

Y = df['Sales']  # Target - Extract the 'Sales' column from the DataFrame

# Define the bin edges and labels
bins = [0, 5, 10, 17]  # Define your own bin edges as needed
labels = ['Low', 'Medium', 'High']  # Define labels for each bin

# Create a new column 'SalesCategory' using the cut function
Y = pd.cut(Y, bins=bins, labels=labels, include_lowest=True)

# Display the Series 'Y' with the new 'SalesCategory'
print(Y)

# Segregate Numeric and Non-numeric columns

# Numeric columns
numeric_features = X.select_dtypes(exclude = ['object']).columns
numeric_features

# Non-numeric columns
categorical_features = X.select_dtypes(include = ['object']).columns
categorical_features 

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''

X.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.5) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['CompPrice', 'Price'])

X_winsor = winsor.fit(X[['CompPrice', 'Price']])

# Save the winsorizer model 
joblib.dump(X_winsor, 'winsorized')

X[['CompPrice', 'Price']] = X_winsor.transform(X[['CompPrice', 'Price']])

X.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
plt.subplots_adjust(wspace = 0.5) 
plt.show()

# MinMaxScaler to convert the magnitude of the columns to a range of 0 to 1
num_pipeline = Pipeline([('scale', MinMaxScaler())])
num_pipeline

# Encoding Non-numeric fields
# **Convert Categorical data  to Numerical data using OneHotEncoder**
categ_pipeline = Pipeline([('OnehotEncode', OneHotEncoder(sparse_output = False))])
categ_pipeline

# Using ColumnTransfer to transform the columns of an array or pandas DataFrame. 
# This estimator allows different columns or column subsets of the input to be
# transformed separately and the features generated by each transformer will
# be concatenated to form a single feature space.
preprocess_pipeline = ColumnTransformer([('categorical', categ_pipeline, categorical_features), 
                                       ('numerical', num_pipeline, numeric_features)], 
                                        remainder = 'passthrough') # Skips the transformations for remaining columns

preprocess_pipeline

# Pass the raw data through pipeline
processed = preprocess_pipeline.fit(X) 

# Save the processed pipeline Model
joblib.dump(processed, 'Processed')

final_data = pd.DataFrame(processed.transform(X), columns = processed.get_feature_names_out())
final_data

X_Train, X_Test, Y_Train, Y_Test = train_test_split(final_data, Y, test_size = 0.2, stratify = Y, random_state = 0)


# Proportion of Target variable categories are consistent across train and test
print(Y_Train.value_counts()/ 800)
print("\n")
print(Y_Test.value_counts()/ 200)

rf_Model = RandomForestClassifier()

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
n_estimators

# Number of features to consider at every split
max_features = ['auto', 'log2']

# Maximum number of levels in tree
max_depth = [2, 7]

# Minimum number of samples required to split a node
min_samples_split = [2, 5]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 3]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(param_grid)

# Hyperparameter optimization with RandomizedSearchCV

rf_Random = RandomizedSearchCV(estimator = rf_Model, param_distributions = param_grid, cv = 10, verbose = 0, n_jobs = -1)

rf_Random.fit(X_Train, Y_Train)

rf_Random.best_params_

cv_rf_random = rf_Random.best_estimator_

# Evaluation on Test Data
test_pred_random = cv_rf_random.predict(X_Test)

accuracy_test_random = np.mean(test_pred_random == Y_Test)
accuracy_test_random

print (f'Train Accuracy - : {rf_Random.score(X_Train, Y_Train):.3f}')
print (f'Test Accuracy - : {rf_Random.score(X_Test, Y_Test):.3f}')

# Plot visualization
cm = skmet.confusion_matrix(Y_Test, test_pred_random)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['High', 'Medium', 'Low'])
cmplot.plot()
cmplot.ax_.set(title = 'Sales Prediction - Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')

# Save the best model from Randomsearch CV approach
pickle.dump(cv_rf_random, open('rfc.pkl', 'wb'))


def cross_validation(model, _X, _y, _cv = 4):
    
    '''Function to perform 4 Folds Cross-Validation
    Parameters
    ----------
    model: Python Class, default=None
          This is the machine learning algorithm to be used for training.
    _X: array
       This is the matrix of features.
    _y: array
       This is the target variable.
    _cv: int, default=5
      Determines the number of folds for cross-validation.
    Returns
    -------
    The function returns a dictionary containing the metrics 'accuracy', 'precision',
    'recall', 'f1' for both training set and validation set.
    '''
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                           X = _X,
                           y = _y,
                           cv = _cv,
                           scoring = _scoring,
                           return_train_score = True)

    return pd.DataFrame({"Training Accuracy scores": results['train_accuracy'],
          "Mean Training Accuracy": results['train_accuracy'].mean()*100,
          "Training Precision scores": results['train_precision'],
          "Mean Training Precision": results['train_precision'].mean(),
          "Training Recall scores": results['train_recall'],
          "Mean Training Recall": results['train_recall'].mean(),
          "Training F1 scores": results['train_f1'],
          "Mean Training F1 Score": results['train_f1'].mean(),
          "Validation Accuracy scores": results['test_accuracy'],
          "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
          "Validation Precision scores": results['test_precision'],
          "Mean Validation Precision": results['test_precision'].mean(),
          "Validation Recall scores": results['test_recall'],
          "Mean Validation Recall": results['test_recall'].mean(),
          "Validation F1 scores": results['test_f1'],
          "Mean Validation F1 Score": results['test_f1'].mean()
          })

Random_forest_result = cross_validation(cv_rf_random, X_Train, Y_Train, 4)

Random_forest_result

def plot_result(x_label, y_label, plot_title, train_data, val_data):
        '''Function to plot a grouped bar chart showing the training and validation
          results of the ML model in each fold after applying K-fold cross-validation.
         Parameters
         ----------
         x_label: str, 
            Name of the algorithm used for training e.g 'Decision Tree'
          
         y_label: str, 
            Name of metric being visualized e.g 'Accuracy'
         plot_title: str, 
            This is the title of the plot e.g 'Accuracy Plot'
         
         train_result: list, array
            This is the list containing either training precision, accuracy, or f1 score.
        
         val_result: list, array
            This is the list containing either validation precision, accuracy, or f1 score.
         Returns
         -------
         The function returns a Grouped Barchart showing the training and validation result
         in each fold.
        '''
        
        # Set size of plot
        plt.figure(figsize=(12, 6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold"]
        X_axis = np.arange(len(labels))
        plt.ylim(0.40000, 1)
        plt.bar(X_axis - 0.2, train_data, 0.1, color = 'blue', label = 'Training')
        plt.bar(X_axis + 0.2, val_data, 0.1, color = 'red', label = 'Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()
        
model_name = "RandomForestClassifier"
plot_result(model_name,
            "Accuracy",
            "Accuracy scores in 4 Folds",
            Random_forest_result["Training Accuracy scores"],
            Random_forest_result["Validation Accuracy scores"])