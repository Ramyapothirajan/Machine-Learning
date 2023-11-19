# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 21:28:30 2023

@author: RamyaRajaLakshmi
"""
'''
# # Clustering

# Problem Statement

# The average retention rate in the insurance industry is 84%, with the 
# top-performing agencies in the 93%-95% range. 
# Retaining customers is all about the long-term relationship you build.
# Offer a discount on the client's current policy will ensure he/she buys a 
# new product or renews the current policy.
# Studying clients' purchasing behavior to figure out which types of products 
# they're most likely to buy is very essential. 

# Insurance company wants to analyze their customer's behaviour to 
# device offers to increase customer loyalty.

# CRISP-ML(Q) process model describes six phases:
# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Model Evaluation
# 5. Deployment
# 6. Monitoring and Maintenance
'''

# Objective(s): Maximize the Sales 
# Constraints: Minimize the Customer Retention

'''Success Criteria'''

# Business Success Criteria: Increase the Sales by 10% to 12% by targeting
# cross-selling opportunities for current customers.

# ML Success Criteria: Achieve a Silhouette coefficient of at least 0.6

# Economic Success Criteria: The insurance company will see an increase in 
# revenues by at least 8%

'''
# Steps to Consider:

Load the Data
Perform EDA and derive insights
Perform Data Preprocessing (steps as appropriate)
Implement Clustering (Hierarchical)
Evaluate the Model - perform tuning by altering hyperparameters - (Silhouette coefficient)
Cluster the Data using the best model (based on the metrics evaluated)
Label the clusters

'''

'''
Data Collection

# Data Dictionary:
    1. Dataset contains 9134 customer details.
    2. 23 features are recorded for each customer 

# Description:
    Employment Status - The current employment status of the customer
    Income - Represents the annual income of the customer
    Customer Lifetime Value - The projected total revenue an insurance company estimates it will earn from a customer throughout their entire relationship.
    Monthly Premium Auto - The amount the customer pays each month as a premium for their auto insurance policy.
    Months Since Last Claim - The number of months that have passed since the customer's most recent insurance claim.
    Months Since Policy Inception - The number of months that have elapsed since the customer's insurance policy was initiated. 
    Total Claim Amount - The total amount claimed by the customer for all insurance claims made.
    Customer - specifies the Customer ID
    State - Location (state) of the  Customer 
    Response - Customer's specific action for offer
    Coverage - The level of coverage or insurance plan the customer has
    Education - The highest level of education attained by the customer
    Effective To Date - The date when the insurance policy becomes effective
    Gender -  Specifies the gender of the customer
    Location Code - The code or category for the customer's location
    Marital Status- The current marital status of the customer
    Number of Open Complaints - The total number of unresolved complaints that the customer has made to the insurance company.
    Number of Policies- The total number of insurance policies held by the customer.
    Policy Type - Category of the insurance policy
    Policy - The specific policy identifier associated with the customer's insurance coverage.
    Renew Offer Type - Type of offer made to the customer to encourage policy renewal.
    Sales Channel - The channel or method through which the policy was sold or marketed.
    Vehicle Class - The class or type of vehicle insured.
    Vehicle Size - The size or category of the insured vehicle.

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sqlalchemy import create_engine
import warnings
import joblib
import dtale

# Ignore warnings
warnings.filterwarnings("ignore")

# **Import the data**
uni = pd.read_csv(r"C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/1. Hierarchical Clustering/Mod3a.Hierarchical Clustering/AutoInsurance.csv")

# Credentials to connect to Database
user = 'root'  # user name
pw = 'Mypassword_23'  # password
db = 'univ_db'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# to_sql() - function to push the dataframe onto a SQL table.
uni.to_sql('univ_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from univ_tbl;'
df = pd.read_sql_query(sql, engine)

# Check the general structure of the data frame
df.info()

# EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS
# ***Descriptive Statistics and Data Distribution Function***

# Data Preprocessing
 
# **Selecting Important Feature**
df1 = df.drop(["Customer", "Effective To Date", "Number of Open Complaints"], axis = 1)
# The selected features aim to capture customer value, risk, and engagement, making them suitable candidates for a machine learning analysis aimed at understanding and predicting customer behavior in the context of insurance.

# Check the general structure of the data frame
df1.info()
# Check if there are any null / missing values in all the columns
df.isna().sum()

df1.head()

df1.describe()

# null / missing values in all the columns in percentage 
duplicate = df.duplicated()  # Returns Boolean Series denoting duplicate rows.
print(duplicate)

sum(duplicate)

# AutoEDA
#  Help to quickly identify patterns, trends, anomalies, and relationships within the data

d = dtale.show(df1)
d.open_browser()

# EDA report highlights:
# ------------------------
# strip before and after whitespaces from all the rows and columns
# apply strip to all rows
df1 = df1.apply(lambda x : x.str.strip() if x.dtype == 'object' else x)
# apply strip to all column names
df1.columns = df1.columns.str.strip()

print(df1['EmploymentStatus'].value_counts())

def categorize_employment_status(x):
    if x == 'Employed' or x == 'Medical Leave':
        return 'Employed'
    elif x == 'Unemployed':
        return 'Unemployed'
    else:
        return 'Other'

df1['EmploymentStatus'] = df1['EmploymentStatus'].apply(categorize_employment_status)

# SVD can be implemented only on Numeric features
numerical_features = df1.select_dtypes(exclude = ['object']).columns
numerical_features

# Outliers: Data points that significantly deviate from the rest of the data and can have a disproportionate impact on statistical analysis and machine learning models.
# Detected exceptional values in 4 columns: Customer Lifetime Value, Monthly Premium Auto, Total Claim Amount
# Boxplot
# Below code ensures each column gets its own y-axis.
# pandas plot() function with parameters kind = 'box' and subplots = True
df1[numerical_features].plot(kind = 'box', subplots = True, sharey = False, figsize = (20,10))
#plt.subplots_adjust(wspace = 0.5)
plt.title("Box Plots Before Winsorization")
plt.tight_layout()
plt.show()

# Winsorization is a data preprocessing technique used to handle outliers in a dataset
# Winsorization involves limiting extreme values by replacing them with values from a certain percentile or a fixed threshold
winsor = Winsorizer(capping_method = 'quantiles', tail = 'both', fold = 0.1, variables = ['Customer Lifetime Value', 'Number of Policies', 'Monthly Premium Auto', 'Total Claim Amount'])
df1_winsorized = winsor.fit_transform(df1[numerical_features])
df1_winsorized = pd.DataFrame(df1_winsorized, columns = numerical_features)
df1_winsorized = pd.concat([df1_winsorized], axis=1)

df1_winsorized.plot(kind = 'box', subplots = True, sharey = False, figsize = (20,10))
#plt.subplots_adjust(wspace = 0.5)
plt.title("Box Plots After treating Outliers")
plt.tight_layout()
plt.show()


# Define SVD model
svd = TruncatedSVD(n_components = 7)

# Define Pipeline to deal with scaling numeric columns
Num_pipeline = make_pipeline(MinMaxScaler(), svd)
Num_pipeline

# Pass the raw data through pipeline
Processed_data = Num_pipeline.fit(df1_winsorized)
Processed_data

# Apply the pipeline on the dataset
Auto_Insurance = pd.DataFrame(Processed_data.transform(df1_winsorized))
Auto_Insurance

# Save the transformed data using joblib
joblib.dump(Processed_data, 'processed_data.pkl')

# Load the transformed data back
loaded_data = joblib.load('processed_data.pkl')

# Apply the saved model on to the Dataset to extract PCA values
data = pd.DataFrame(loaded_data.transform(df1_winsorized))
data.head()

# SVD weights
svd.components_

# Take a closer look at the components
components = pd.DataFrame(loaded_data['svd'].components_, columns = numerical_features).T
components.columns = ['SVD_1', 'SVD_2', 'SVD_3', 'SVD_4', 'SVD_5', 'SVD_6', 'SVD_7']

components

# Create a heatmap to visualize the SVD component loadings
svd_components = svd.components_
svd_components_df = pd.DataFrame(svd_components, columns=df1_winsorized.columns)
plt.figure(figsize=(10, 6))
sns.heatmap(svd_components_df, annot=True, cmap='coolwarm')
plt.title('SVD Component Loadings')
plt.show()

# Scree Plot for SVD
explained_variance_svd = svd.singular_values_ ** 2  # Squaring the singular values
explained_variance_ratio_svd = explained_variance_svd / np.sum(explained_variance_svd)
cumulative_variance_ratio_svd = np.cumsum(explained_variance_ratio_svd)

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_svd) + 1), explained_variance_ratio_svd, label='Explained Variance Ratio')
plt.plot(range(1, len(explained_variance_svd) + 1), cumulative_variance_ratio_svd, marker='o', label='Cumulative Explained Variance Ratio')
plt.xlabel('Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot: Explained Variance Ratio and Cumulative Explained Variance (SVD)')
plt.legend()

# Identify the knee point visually and set the knee_point variable accordingly
knee_point_svd = 4 # Adjust this value based on your observation
plt.axvline(x=knee_point_svd, color='red', linestyle='--', label='Knee Point')
plt.legend()

plt.show()

# Scree Plot recommends 5 PCA's as the ideal number of features to be considered
# The goal in choosing the number of principal components is to retain a sufficient amount of the total variance in your data while reducing dimensionality. 
# The cumulative explained variance ratio after the 5th component is 0.8604 (86.04%).

# Final dataset with manageable number of columns (Feature Extraction)
final_result = pd.concat([df1.State, df1['EmploymentStatus'], df1['Coverage'],df1['Renew Offer Type'], df1['Policy'],  data.iloc[:, 0:4]], axis = 1)

# Define a dictionary to map new names to specific columns
column_rename_dict = {
    0: 'SVD_1',
    1: 'SVD_2',
    2: 'SVD_3',
    3: 'SVD_4',
    4: 'SVD_5'
}
# Rename the specified columns
final_result.rename(columns=column_rename_dict, inplace=True)

# Display the DataFrame with renamed columns
print(final_result.head())

# List of SVD component pairs
svd_pairs = [('SVD_1', 'SVD_2'), ('SVD_1', 'SVD_3'), ('SVD_1', 'SVD_4'),
             ('SVD_2', 'SVD_3'), ('SVD_2', 'SVD_4'), ('SVD_3', 'SVD_4')]

# Create a subplot grid
num_rows = 2
num_cols = 3
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))
# Loop through each SVD pair and create scatter plots in subplots
for (x_pca, y_pca), ax in zip(svd_pairs, axes.flatten()):
    ax.scatter(final_result[x_pca], final_result[y_pca], alpha=0.5)
    ax.set_xlabel(x_pca)
    ax.set_ylabel(y_pca)
    ax.set_title(f'Scatter Plot: {x_pca} vs {y_pca}')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()