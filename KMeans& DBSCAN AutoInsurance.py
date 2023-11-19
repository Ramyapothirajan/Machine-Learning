# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 20:49:58 2023

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
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from clusteval import clusteval 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
import pickle, joblib
import dtale

# Ignore warnings
warnings.filterwarnings("ignore")

# Read the source file into data frame. Set SkipInitialspace = True to remove extra white spaces
df = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/1. Hierarchical Clustering/Mod3a.Hierarchical Clustering/AutoInsurance.csv", skipinitialspace = True)

# Check the general structure of the data frame
df.info()

df2 = df
df2['Effective To Date'] = pd.to_datetime(df['Effective To Date'])
df2['Day of Week'] = df2['Effective To Date'].dt.dayofweek
df2['Year'] = df2['Effective To Date'].dt.year
df2['Month'] = df2['Effective To Date'].dt.month
df2['Day'] = df2['Effective To Date'].dt.day

# Grouping by year and calculating average values for numerical columns
average_by_year = df2.groupby('Year')[['Monthly Premium Auto', 'Months Since Last Claim', 'Months Since Policy Inception', 'Number of Policies']].mean()
# Grouping by month and calculating average values for numerical columns
average_by_month = df2.groupby('Month')[['Monthly Premium Auto', 'Months Since Last Claim','Months Since Policy Inception', 'Number of Policies']].mean()
# Grouping by day and calculating average values for numerical columns
average_by_day = df2.groupby('Day')[['Monthly Premium Auto', 'Months Since Last Claim', 'Months Since Policy Inception', 'Number of Policies']].mean()

# Plotting trends by month
average_by_month.plot(kind='line', figsize=(10, 6))
plt.xlabel('Month')
plt.ylabel('Average Value')
plt.title('Average Trends by Month')
plt.show()

# Plotting trends by day
average_by_day.plot(kind='line', figsize=(10, 6))
plt.xlabel('Day')
plt.ylabel('Average Value')
plt.title('Average Trends by Day')
plt.show()

# EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS
# ***Descriptive Statistics and Data Distribution Function***

# Data Preprocessing

# **Selecting Important Feature**

# The selected features aim to capture customer value, risk, and engagement, making them suitable candidates for a machine learning analysis aimed at understanding and predicting customer behavior in the context of insurance.

selected_columns = ['Customer Lifetime Value', 'Coverage', 'Number of Policies', 'EmploymentStatus', 'Income', 'Monthly Premium Auto', 'Months Since Last Claim', 'Months Since Policy Inception', 'Total Claim Amount']
df1 = df[selected_columns]
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

# Most of the ML algorithms doesn't like categorical variables, hence we need to ensure we convert all the feature values into numeric before proceeding to analysis part
# Segregate Numeric and Non-numeric columns 

numerical_features = df1.select_dtypes(exclude = ['object']).columns
numerical_features

categorical_features = df1.select_dtypes(include = ['object']).columns
categorical_features

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
df1_winsorized = pd.concat([df1_winsorized, df1[categorical_features]], axis=1)

df1_winsorized.plot(kind = 'box', subplots = True, sharey = False, figsize = (20,10))
#plt.subplots_adjust(wspace = 0.5)
plt.title("Box Plots After treating Outliers")
plt.tight_layout()
plt.show()

# Define Pipeline to deal with scaling numeric columns
Num_pipeline = Pipeline([('sclaer', Normalizer())])
Num_pipeline

# Define Pipeline to deal with encoding categorical columns
Categ_pipeline = Pipeline(([('OnehotEncode', OneHotEncoder(drop = 'first'))]))
Categ_pipeline

# Using ColumnTransfer to transform the columns of an array or pandas DataFrame. 
# This estimator allows different columns or column subsets of the input to be transformed separately
# The features generated by each transformer will be concatenated to form a single feature space.
Processed_pipeline = ColumnTransformer([('categorical', Categ_pipeline, categorical_features),
('numerical', Num_pipeline, numerical_features)])

Processed_pipeline

# Fit and transform the data using Processed_pipeline
Processed = Processed_pipeline.fit_transform(df1_winsorized)

# Save the transformed data using joblib
joblib.dump(Processed, 'Processed')

# Load the transformed data back into a DataFrame 'data'
transformed_data = joblib.load('Processed')
data = pd.DataFrame(transformed_data, columns = list(Processed_pipeline.get_feature_names_out()))

data.head()

data.describe()

# =============================== Model Building =========================================
# CLUSTERING MODEL BUILDING

###### scree plot or elbow curve ############
# TWSS is a term used in the context of analysis of variance (ANOVA) and regression analysis to quantify the total variation in a dataset. 
# It represents the sum of the squared differences between each data point and the overall mean of the data.
TWSS = []
k = list(range(2, 11))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data)
    TWSS.append(kmeans.inertia_)

TWSS

# ## Creating a scree plot to find out no.of cluster
plt.plot(k, TWSS, 'go-.'); plt.xlabel("No of Clusters"); plt.ylabel("Total within Sum of Squares")
plt.title("TWSS vs No of Clusters")
plt.show()

# Building KMeans clustering
model = KMeans(n_clusters = 2)
output = model.fit(data)

# Analysing the output by referring to the cluster labels assigned
model.labels_

# Clusters Evaluation
# Silhouette Score : Measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). 
# The score ranges from -1 to +1, where higher values indicate better-defined clusters.
metrics.silhouette_score(data, model.labels_)

# Calinski-Harabasz Index: The Variance Ratio Criterion, this index measures the ratio of between-cluster variance to within-cluster variance.
# Higher values indicate well-separated clusters.
metrics.calinski_harabasz_score(data, model.labels_)

# Davies-Bouldin Index: Measures the average similarity between each cluster and its most similar cluster. 
# Lower values indicate better-defined clusters.
metrics.davies_bouldin_score(data, model.labels_)

# ================== Clusters Evaluation ============================================
'''Hyperparameter Optimization for Hierarchical Clustering'''
# Experiment to obtain the best clusters by altering the parameters
# Building KMeans clustering
finalmodel = KMeans(n_clusters = 4)
output = finalmodel.fit(data)

# Save the KMeans Clustering Model
pickle.dump(output, open('Auto_Insurance.pkl', 'wb'))

import os
os.getcwd()

# Clusters Evaluation
# Silhouette Score 
metrics.silhouette_score(data, finalmodel.labels_)

# Calinski-Harabasz Index: The Variance Ratio Criterion, this index measures the ratio of between-cluster variance to within-cluster variance.
# Higher values indicate well-separated clusters.
metrics.calinski_harabasz_score(data, finalmodel.labels_)

# Davies-Bouldin Index: Measures the average similarity between each cluster and its most similar cluster. 
# Lower values indicate better-defined clusters.
metrics.davies_bouldin_score(data, finalmodel.labels_)

f = pd.Series(finalmodel.labels_)
df_cluster = pd.concat([f, df.State, df['Coverage'], df['Number of Policies'], df['Renew Offer Type'], df['Vehicle Class'], df1], axis = 1)
df_cluster = df_cluster.rename(columns = {0:'ClusterID'})
df_cluster.head()

# Group by Employment Status, Policy Type and count occurrences of each cluster
Policy_cluster_distribution = df_cluster.groupby([df1['EmploymentStatus'], df['Policy Type'],  df_cluster['ClusterID']]).size()
# Print the state-cluster distribution
print(Policy_cluster_distribution)

# Reshape the data to create a pivot table for the stacked bar plot
pivot_table = Policy_cluster_distribution.unstack(fill_value=0)
   
# Create a single stacked bar plot for all 'EmploymentStatus' categories
pivot_table.plot(kind='bar', stacked=True, figsize=(12, 8))

plt.title('Cluster Distribution by EmploymentStatus')
plt.xlabel('EmploymentStatus - Policy Type')
plt.ylabel('Count')
plt.legend(title='Cluster', bbox_to_anchor=(1, 1))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Group by State and count occurrences of each cluster
State_cluster_distribution = df_cluster.groupby([df['State'], df_cluster['ClusterID']]).size()
print(State_cluster_distribution)

# Reshape the data to create a pivot table for the stacked bar plot
pivot_table = State_cluster_distribution.unstack(fill_value=0)

# Create a stacked bar plot for each state
states = pivot_table.index

# Create a single stacked bar plot for all states
pivot_table.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Cluster Distribution by State')
plt.xlabel('State')
plt.ylabel('Count')
plt.legend(title='Cluster', bbox_to_anchor=(1, 1))
plt.xticks(rotation=0)
plt.tight_layout()

# Group by State and count occurrences of each cluster
Coverage_cluster_distribution = df_cluster.groupby([df_cluster['ClusterID'],df['Coverage']]).size()
print(Coverage_cluster_distribution)

# Group the data by 'Cluster' and calculate the mean for specific columns
# Aggregate using the mean of each cluster
result = df_cluster.iloc[:,:].groupby(df_cluster.ClusterID).mean()
print(result)

# Combine the cluster labels with the original data
clust_out = pd.concat([df.State, df['Policy Type'], df['Renew Offer Type'], df['Vehicle Class'] , f, df1_winsorized], axis = 1)
clust_out = clust_out.rename(columns = {0 : 'ClusterID'})

# Get the 'Cluster' and 'EmploymentStatus' columns
cluster_column = clust_out.pop('ClusterID')
policy_type_column = clust_out.pop('Policy Type')
coverage_column = clust_out.pop('Coverage')
employment_status_column = clust_out.pop('EmploymentStatus')

# Insert the columns at the desired indices
clust_out.insert(0, 'ClusterID', cluster_column)
clust_out.insert(2, 'EmploymentStatus', employment_status_column)
clust_out.insert(3, 'Policy Type', policy_type_column)
clust_out.insert(4, 'Coverage', coverage_column)

clust_out.to_csv('AutoInsurance_updated.csv', encoding = 'utf-8')   

import os
os.getcwd()

#=================== DBSCAN Clustering ================================

# Apply DBSCAN clustering
db_scan = DBSCAN(eps = 1.0, min_samples=5)  # Adjust these parameters
db_clust = db_scan.fit_predict(data)

# Check for valid number of clusters
unique_labels = set(db_clust)
num_clusters = len(unique_labels)
if num_clusters >= 3:
    silhouette_avg = silhouette_score(data, db_clust)
    print("Number of Clusters:", num_clusters)
    print("Silhouette Score:", silhouette_avg)
else:
    print("Insufficient clusters for silhouette score calculation.")

# Print the cluster assignments
print("Cluster Assignments:", db_clust)

pickle.dump(db_scan, open('db_scan.pkl', 'wb'))

model = pickle.load(open('db_scan.pkl', 'rb'))

model.labels_

new_cluster_labels = pd.Series(model.labels_)
final_cluster = pd.concat([new_cluster_labels, df1_winsorized], axis = 1)
final_cluster = final_cluster.rename(columns = {0 : 'Cluster'})

# Now data_with_clusters contains your original data along with the cluster assignments
print(final_cluster.head())

# Group by State and count occurrences of each cluster
State_cluster_distribution = final_cluster.groupby([df['State'], final_cluster['Cluster']]).size()
print(State_cluster_distribution)

# Reshape the data to create a pivot table for the stacked bar plot
pivot_table = State_cluster_distribution.unstack(fill_value=0)

# Create a stacked bar plot for each state
states = pivot_table.index

# Create a single stacked bar plot for all states
pivot_table.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Cluster Distribution by State')
plt.xlabel('State')
plt.ylabel('Count')
plt.legend(title='Cluster', bbox_to_anchor=(1, 1))
plt.xticks(rotation=0)
plt.tight_layout()

# Group by Employment Status, Policy Type and count occurrences of each cluster
Policy_cluster_distribution = final_cluster.groupby([df1['EmploymentStatus'],  final_cluster['Cluster']]).size()
# Print the state-cluster distribution
print(Policy_cluster_distribution)

# Group the data by 'Cluster' and calculate the mean for specific columns
# Aggregate using the mean of each cluster
result = final_cluster.iloc[:,:].groupby(final_cluster.Cluster).mean()
print(result)
