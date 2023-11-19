## Problem Statement
'''
An analytics company has been tasked with the crucial job of finding out what factors affect a startup company.

#### No Pre-Set Standards

`What can be the major factors that affect a startup company?`

#### Revolutionizing the startup company through Machine Learning

**Linear regression**
Linear regression is a ML model that estimates the relationship between independent variables and a dependent variable using a linear equation (straight line equation) in a multidimensional space.

**CRISP-ML(Q) process model describes six phases:**

- Business and Data Understanding
- Data Preparation (Data Engineering)
- Model Building (Machine Learning)
- Model Evaluation and Tunning
- Deployment
- Monitoring and Maintenance


**Objective(s):** Maximize the profits

**Constraints:** Maximize the stakeholder satisfaction

**Success Criteria**

- **Business Success Criteria**: Improve the profits from anywhere between 10% to 20%

- **ML Success Criteria**: RMSE should be less than 0.15

- **Economic Success Criteria**: Increase in profits & accuarcy by atleast 20 to 30%
'''
# Load the Data and perform EDA and Data Preprocessing

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sidetable
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from feature_engine.outliers import Winsorizer

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import joblib
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

df = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/18. Multiple Linear Regression/50_Startups.csv")

# Change column names
df.columns = ['R&D', 'Administration', 'Marketing', 'State', 'Profit']

#### Descriptive Statistics and Data Distribution
df.describe()

# Missing values check
df.isnull().any()
df.info()


# Seperating input and output variables 
X = pd.DataFrame(df.iloc[:, 0:4])
y = pd.DataFrame(df.iloc[:, 4])

# Checking for unique values & Counts
X["State"].value_counts()

# Build a frequency table using sidetable library
X.stb.freq(["State"])

# Segregating Non-Numeric features
categorical_features = X.select_dtypes(include = ['object']).columns
print(categorical_features)


# Segregating Numeric features
numeric_features = X.select_dtypes(exclude = ['object']).columns
print(numeric_features)


## Missing values Analysis
# Define pipeline for missing data if any

num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean'))])

preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features)])

# Fit the imputation pipeline to input features
imputation = preprocessor.fit(X)

# Save the pipeline
joblib.dump(imputation, 'meanimpute')

# Transformed data
cleandata = pd.DataFrame(imputation.transform(X), columns = numeric_features)
cleandata


## Outlier Analysis

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True

X.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 

'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''

# Increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


# Winsorization for outlier treatment
winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = list(cleandata.columns))

winsor

clean = winsor.fit(cleandata)

# Save winsorizer model
joblib.dump(clean, 'winsor')

cleandata1 = pd.DataFrame(clean.transform(cleandata), columns = numeric_features)

# Boxplot
cleandata1.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


# Scaling
## Scaling with MinMaxScaler
scale_pipeline = Pipeline([('scale', MinMaxScaler())])

scale_columntransfer = ColumnTransformer([('scale', scale_pipeline, numeric_features)]) # Skips the transformations for remaining columns

scale = scale_columntransfer.fit(cleandata1)

# Save Minmax scaler pipeline model
joblib.dump(scale, 'minmax')

scaled_data = pd.DataFrame(scale.transform(cleandata1), columns = numeric_features)
scaled_data.describe()

## Encoding
# Categorical features
encoding_pipeline = Pipeline([('onehot', OneHotEncoder())])

preprocess_pipeline = ColumnTransformer([('categorical', encoding_pipeline, categorical_features)])

clean =  preprocess_pipeline.fit(X)   # Works with categorical features only

# Save the encoding model
joblib.dump(clean, 'encoding')

encode_data = pd.DataFrame(clean.transform(X))


# To get feature names for Categorical columns after Onehotencoding 
encode_data.columns = clean.get_feature_names_out(input_features = X.columns)
encode_data.info()

clean_data = pd.concat([scaled_data, encode_data], axis = 1)  # concatenated data will have new sequential index
clean_data.info()


####################
# Multivariate Analysis
sns.pairplot(df)   # original data

# Correlation Analysis on Original Data
orig_df_cor = df.corr()
orig_df_cor

# Colinearity pairs observed:
    # R&D - Profit: 0.973
    # R&D - Marketing : 0.724
    # Profit - Marketing : 0.747

# Heatmap
dataplot = sns.heatmap(orig_df_cor, annot = True, cmap = "YlGnBu")

# Heatmap enhanced
# Generate a mask to show values on only the bottom triangle
# Upper triangle of an array.
mask = np.triu(np.ones_like(orig_df_cor, dtype = bool))
sns.heatmap(orig_df_cor, annot = True, mask = mask, vmin = -1, vmax = 1)
plt.title('Correlation Coefficient Of Predictors')
plt.show()

# Library to call OLS model
# import statsmodels.api as sm

# Build a vanilla model on full dataset

# By default, statsmodels fits a line passing through the origin, i.e. it 
# doesn't fit an intercept. Hence, you need to use the command 'add_constant' 
# so that it also fits an intercept

P = add_constant(clean_data)
basemodel = sm.OLS(y, P).fit()
basemodel.summary()

# p-values of coefficients found to be insignificant due to colinearity

# Identify the variale with highest colinearity using Variance Inflation factor (VIF)
# Variance Inflation Factor (VIF = 1 / (1-R2))
# Assumption: VIF > 10 = colinearity
# VIF on clean Data
vif = pd.Series([variance_inflation_factor(P.values, i) for i in range(P.shape[1])], index = P.columns)
vif
# All values of vif < 10

# Tune the model by verifying for influential observations
sm.graphics.influence_plot(basemodel)

clean_data1 = clean_data.drop(clean_data.index[[48,49]])
y_new = y.drop(y.index[[48,49]])

# Build model on dataset
basemode1 = sm.OLS(y_new, clean_data1).fit()
basemode1.summary()

# Regularization Techniques: LASSO, RIDGE and Elasticnet Regression
### LASSO MODEL ###

lasso = Lasso(alpha = 0.10)

lasso.fit(clean_data1, y_new)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(clean_data1.columns))

# Create a function called lasso,
pred_lasso = lasso.predict(clean_data1)

# Adjusted r-square
s1 = lasso.score(clean_data1, y_new)
s1

# RMSE
np.sqrt(np.mean((pred_lasso - np.array(y_new['Profit']))**2))

### RIDGE REGRESSION ###
rm = Ridge(alpha = 0.12)

rm.fit(clean_data1, y_new)

# Coefficients values for all the independent vairbales
rm.coef_
result = rm.coef_.flatten()
rm.intercept_

plt.bar(height = pd.Series(result), x = pd.Series(clean_data1.columns))

rm.alpha

pred_rm = rm.predict(clean_data1)

# Adjusted r-square
s2 = rm.score(clean_data1, y_new.Profit)
s2
# RMSE
np.sqrt(np.mean((pred_rm - np.array(y_new['Profit']))**2))


### ELASTIC NET REGRESSION ###
enet = ElasticNet(alpha = 0.12)

enet.fit(clean_data1, y_new.Profit) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(clean_data1.columns))

enet.alpha

pred_enet = enet.predict(clean_data1)

# Adjusted r-square
s3 = enet.score(clean_data1, y_new.Profit)
s3
# RMSE
np.sqrt(np.mean((pred_enet - np.array(y_new.Profit))**2))


####################
# Lasso Regression
lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(clean_data1, y_new.Profit)

lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(clean_data1)

# Adjusted r-square#
s4 = lasso_reg.score(clean_data1, y_new.Profit)
s4
# RMSE
np.sqrt(np.mean((lasso_pred - np.array(y_new.Profit))**2))


# Ridge Regression
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import Ridge

ridge = Ridge()
# parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(clean_data1, y_new.Profit)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(clean_data1)

# Adjusted r-square#
s5 = ridge_reg.score(clean_data1, y_new.Profit)
s5
# RMSE
np.sqrt(np.mean((ridge_pred - np.array(y_new.Profit))**2))


# ElasticNet Regression
enet = ElasticNet()

# parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'r2', cv = 5)

enet_reg.fit(clean_data1, y_new.Profit)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(clean_data1)

# Adjusted r-square
s6 = enet_reg.score(clean_data1, y_new.Profit)
s6

# RMSE
np.sqrt(np.mean((enet_pred - np.array(y_new.Profit))**2))

scores_all = pd.DataFrame({'models':['Lasso', 'Ridge', 'Elasticnet', 'Grid_lasso', 'Grid_ridge', 'Grid_elasticnet'], 'Scores':[s1, s2, s3, s4, s5, s6]})
scores_all

'Best score obtained for Lasso,  Gridsearch Ridge, Gridsearch Elasticnet Regression'

finalgrid1 = lasso.best_estimator_
finalgrid2 = enet_reg.best_estimator_
finalgrid3 = ridge_reg.best_estimator_

# Save the best model
pickle.dump(finalgrid1, open('lasso.pkl', 'wb'))
pickle.dump(finalgrid2, open('grid_enet.pkl', 'wb'))
pickle.dump(finalgrid3, open('grid_ridge.pkl', 'wb'))