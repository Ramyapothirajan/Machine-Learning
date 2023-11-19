'''Simple Linear regression
# Simple linear regression is a regression model that estimates the relationship between one independent variable and a dependent variable using a straight line.

Problem Statement

The time taken for delivery – time taken for the sorting of the items for delivery Relationship:

# linear relationship between the independent variable (sorting time) and the dependent variable (delivery time).

# Understanding and analyzing the discrepancies between delivery and sorting times are crucial for a logistics company to identify potential bottlenecks or inefficiencies in their operations. 
It can help in streamlining the process, optimizing efficiency, and enhancing customer service.

RMSE (Root Mean Squared Error): Measures the average difference between the predicted and observed values. 
                                A lower RMSE indicates better model performance.
                                
Correlation Coefficient: It signifies the strength and direction of the linear relationship between the variables. 
The value ranges from -1 to 1, where 1 represents a perfect positive linear relationship, -1 a perfect negative linear relationship, and 0 indicates no linear relationship.

'''

# CRISP-ML(Q) process model describes six phases:
# 
# - Business and Data Understanding
# - Data Preparation (Data Engineering)
# - Model Building (Machine Learning)
# - Model Evaluation and Tunning
# - Deployment
# - Monitoring and Maintenance
# 
'''
# **Objective(s):** Minimize Delivery Time
# or 
# Minimize Discrepancy Between Delivery and Sorting Time

'''

import pandas as pd # for Data Manipulation
import numpy as np # for Mathematical calculations
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from sklearn.preprocessing import FunctionTransformer
import sweetviz as sv


df = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/17. Simple Linear Regression/SLR_Dataset/delivery_time.csv")

# Change column names
df.columns = ['Delivery_Time', 'Sorting_Time']

df.head()

# Relevant fields for Regression Analysis
df.info()

df.describe()

df.sort_values('Delivery_Time', ascending = True, inplace = True)

df.reset_index(inplace = True, drop = True)
df.head(10)

# Select numeric features for data preprocessing
numeric_features = df['Sorting_Time']

# Box plot - to check if there any outliers 
df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 

''' No Outliers and missing values are there in the given dataset'''

# increase spacing between subplots
plt.subplots_adjust(wspace = 0.6) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

plt.hist(df['Delivery_Time']) #histogram

plt.hist(df['Sorting_Time'])


# Analyzing the dataset
report = sv.analyze(df)
# Display the report
report.show_html('EDAreport.html') # html report generated in working directory


# # Bivariate Analysis
# Scatter plot
plt.scatter(x = df['Sorting_Time'], y = df['Delivery_Time']) 

## Measure the strength of the relationship between two variables using Correlation coefficient.
np.corrcoef(df['Sorting_Time'],df['Delivery_Time'])

# Covariance
cov_output = np.cov(df['Sorting_Time'], df['Delivery_Time'])[0, 1]
cov_output

# df.cov()
dataplot = sns.heatmap(df.corr(), annot = True, cmap = "coolwarm")

# # Linear Regression using statsmodels package
# Simple Linear Regression
# ols - Ordinary Least Square
model = smf.ols('Delivery_Time ~ Sorting_Time', data = df).fit()

model.summary()

pred1 = model.predict(pd.DataFrame(df['Sorting_Time']))

pred1

# Regression Line
plt.scatter(df['Sorting_Time'], df['Delivery_Time'])
plt.plot(df['Sorting_Time'], pred1, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

# Error calculation (error = AV - PV)
res1 = df['Delivery_Time'] - pred1

print(np.mean(res1))

res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

# # Model Tuning with Transformations
# ## Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(df['Sorting_Time']), y = df['Delivery_Time'], color = 'brown')
np.corrcoef(np.log(df['Sorting_Time']), df['Delivery_Time']) #correlation

model2 = smf.ols('Delivery_Time ~ np.log(Sorting_Time)', data = df).fit()
model2.summary()


pred2 = model2.predict(pd.DataFrame(df['Sorting_Time']))

# Regression Line
plt.scatter(np.log(df['Sorting_Time']), df['Delivery_Time'])
plt.plot(np.log(df['Sorting_Time']), pred2, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

# Error calculation
res2 = df['Delivery_Time'] - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

# ## Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = df['Sorting_Time'], y = np.log(df['Delivery_Time']), color = 'orange')
np.corrcoef(df['Sorting_Time'], np.log(df['Delivery_Time'])) #correlation

model3 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time', data = df).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(df['Sorting_Time']))

# Regression Line
plt.scatter(df['Sorting_Time'], np.log(df['Delivery_Time']))
plt.plot(df['Sorting_Time'], pred3, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()


pred3_Delivery_Time = np.exp(pred3)
print(pred3_Delivery_Time)

res3 = df['Delivery_Time'] - pred3_Delivery_Time
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

# ## Polynomial transformation 
# x = waist; x^2 = waist*waist; y = log(at)


X = pd.DataFrame(df['Sorting_Time'])
# X.sort_values(by = ['Waist'], axis = 0, inplace = True)

Y = pd.DataFrame(df['Delivery_Time'])


model4 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time + I(Sorting_Time*Sorting_Time)', data = df).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(df))
print(pred4)


plt.scatter(X['Sorting_Time'], np.log(Y['Delivery_Time']))
plt.plot(X['Sorting_Time'], pred4, color = 'red')
plt.plot(X['Sorting_Time'], pred3, color = 'green', label = 'linear')
plt.legend(['Transformed Data', 'Polynomial Regression Line', 'Linear Regression Line'])
plt.show()

pred4_Delivery_Time = np.exp(pred4)
pred4_Delivery_Time

# Error calculation
res4 = df['Delivery_Time'] - pred4_Delivery_Time
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

### Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)

table_rmse

# # Evaluate the best model
# Data Split
train, test = train_test_split(df, test_size = 0.2, random_state = 0)

plt.scatter(train.Sorting_Time, np.log(train.Delivery_Time))

plt.figure(2)
plt.scatter(test.Sorting_Time, np.log(test.Delivery_Time))

# Fit the best model on train data
finalmodel = smf.ols('Delivery_Time ~ np.log(Sorting_Time)', data = train).fit()


# Predict on test data
test_pred = finalmodel.predict(test)

# Model Evaluation on Test data
test_res = test.Delivery_Time - test_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)

test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))

# Model Evaluation on train data
train_res = train.Delivery_Time - train_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)

train_rmse

##### Save the Best model for Pipelining
# Logarithmic transformation function
log_transform = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False)
log_model = make_pipeline(log_transform, LinearRegression())
log_model.fit(df[['Sorting_Time']], df[['Delivery_Time']])

pickle.dump(log_model, open('log_model.pkl', 'wb'))

# load the saved pipelines
poly_model = pickle.load(open('log_model.pkl', 'rb'))
