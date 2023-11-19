'''
#CRISP-ML(Q) Framework: 

Business Understanding:

Business Problem: A glass manufacturing plant uses different earth elements to design new glass materials based on customer requirements.
They would like to automate the process of classification as it’s a tedious job to manually classify them.Classifying the glass type based on the other features using KNN algorithm. 

Business Objective: Maximize Sales
Business Constraint: Maximize Customer Satisfaction 

Success Criteria:
Business: Effective Marketing
ML: Achieve an accuracy of more than 85%
Economic: Reach profit more than 15%

Data Understanding: 
214 observations & 10 columns

9 columns are inputs & 1 column is output

1. RI: Refractive index.
2. Na: Sodium (content in weight percent).
3. Mg: Magnesium (content in weight percent).
4. Al: Aluminum (content in weight percent).
5. Si: Silicon (content in weight percent).
6. K: Potassium (content in weight percent).
7. Ca: Calcium (content in weight percent).
8. Ba: Barium (content in weight percent).
9. Fe: Iron (content in weight percent).
10.Type: The target variable representing the type or class of glass.      

'''

import os 
import pandas as pd
import matplotlib.pyplot as plt
import sweetviz
import pickle
import joblib
import graphviz
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_engine.outliers import Winsorizer

import sklearn.metrics as skmet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_validate

df = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/12. KNN_Classifier/glass.csv")

df.columns

df.info()

# Checking for Null values
df.isnull().sum()

#AutoEDA
report = sweetviz.analyze([df, "data"])
report.show_html('Report.html')

# Target variable categories
df['Type'].value_counts()

# Data Preprocessing & EDA
# Mentioning 1 as Type_1, 2 as Type_2,..
df['Type'] = df['Type'].replace({1: 'Type_1', 2: 'Type_2', 3: 'Type_3', 5: 'Type_4', 6: 'Type_6', 7: 'Type_7'})

df1 =df.drop("Ba", axis=1) #  ['Ba'] have low variation of values
 
# Data split into Input and Output
X = df1.iloc[:, 0:8] # Predictors
Y = df1['Type']  # Target - Extract the 'Type' column from the DataFrame

# Display the Series 'Y' with the new 'SalesCategory'
print(Y)

# Numeric columns
numeric_features = X.select_dtypes(exclude = ['object']).columns
numeric_features

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
                          variables = ['RI', 'Na', 'Al', 'Si', 'K', 'Ca', 'Fe'])

X_winsor = winsor.fit(X[['RI', 'Na', 'Al', 'Si', 'K', 'Ca', 'Fe']])

# Save the winsorizer model 
joblib.dump(X_winsor, 'winsorized.pkl')

X[['RI', 'Na', 'Al', 'Si', 'K', 'Ca', 'Fe']] = X_winsor.transform(X[['RI', 'Na', 'Al', 'Si', 'K', 'Ca', 'Fe']])

X.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
plt.subplots_adjust(wspace = 0.5) 
plt.show()

# MinMaxScaler to convert the magnitude of the columns to a range of 0 to 1
num_pipeline = Pipeline([('scale', MinMaxScaler())])
num_pipeline

preprocess_pipeline = ColumnTransformer([ ('numerical', num_pipeline, numeric_features)], 
                                        remainder = 'passthrough') # Skips the transformations for remaining columns

preprocess_pipeline

# Pass the raw data through pipeline
processed = preprocess_pipeline.fit(X) 

# Save the processed pipeline Model
joblib.dump(processed, 'Processed')

os.getcwd()

final_data = pd.DataFrame(processed.transform(X), columns = processed.get_feature_names_out())
final_data.describe()

X_Train, X_Test, Y_Train, Y_Test = train_test_split(final_data, Y, test_size = 0.2, stratify = Y, random_state = 0)


# Proportion of Target variable categories are consistent across train and test
print(Y_Train.value_counts()/ 800)
print("\n")
print(Y_Test.value_counts()/ 200)

X_Train.shape
X_Test.shape

# Model building
knn = KNeighborsClassifier(n_neighbors = 15)

KNN = knn.fit(X_Train, Y_Train)  # Train the kNN model

# Evaluate the model with train data
pred_train = knn.predict(X_Train)  # Predict on train data

pred_train

# Cross table
pd.crosstab(Y_Train, pred_train, rownames = ['Actual'], colnames = ['Predictions']) 

print(skmet.accuracy_score(Y_Train, pred_train))  # Accuracy measure

# Predict the class on test data
pred = knn.predict(X_Test)
pred

# Evaluate the model with test data
print(skmet.accuracy_score(Y_Test, pred))
pd.crosstab(Y_Test, pred, rownames = ['Actual'], colnames = ['Predictions']) 

# creating empty list variable 
acc = []

# running KNN algorithm for 2 to 40 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(4, 40, 3):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X_Train, Y_Train)
    train_acc = np.mean(neigh.predict(X_Train) == Y_Train)
    test_acc = np.mean(neigh.predict(X_Test) == Y_Test)
    diff = train_acc - test_acc
    acc.append([diff, train_acc, test_acc])
    
acc
    
# Plotting the data accuracies
plt.plot(np.arange(4, 40, 3), [i[1] for i in acc], "ro-")
plt.plot(np.arange(4, 40, 3), [i[2] for i in acc], "bo-")
# Add a title
plt.title("Model Accuracy for Different Values - KNN")
plt.show()

# Hyperparameter optimization
k_range = list(range(4, 40, 3))
param_grid = dict(n_neighbors = k_range)
  
# Defining parameter range
grid = GridSearchCV(knn, param_grid, cv = 5, 
                    scoring = 'accuracy', 
                    return_train_score = False, verbose = 1)


KNN_new = grid.fit(X_Train, Y_Train) 

print(KNN_new.best_params_)

accuracy = KNN_new.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )

# Predict the class on test data
pred = KNN_new.predict(X_Test)
pred

# Save the model
knn_best = KNN_new.best_estimator_
pickle.dump(knn_best, open('knn.pkl', 'wb'))

cm = skmet.confusion_matrix(Y_Test, pred)

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Type_1', 'Type_2', 'Type_3', 'Type_5', 'Type_6', 'Type_7'])
cmplot.plot()
cmplot.ax_.set(title = 'Glass Material Type Detection - Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')