# # Problem Statement: -
# There are two datasets consisting of information for the connecting routes and flight hault. Create network analytics models on both the datasets separately and measure degree centrality, degree of closeness centrality, and degree of in-between centrality.
# Create a network using edge list matrix(directed only).

# Network analytics is the application of big data principles and tools to the management and security of data networks.


import pandas as pd
import os
import networkx as nx 
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# Creating engine which link to SQL via python
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",# user
                               pw="Password_123", #passwrd
                               db="air_routes_db")) #database

# Define column names as a list
column_names = ["Flights", "ID", "Main Airport", "Main Airport ID", "Destination Airport", "Destination Airport ID", "Haults", "Unnamed", "Machinery"]

# Reading data from local drive
connecting_routes = pd.read_csv(r"C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/8. Network Analytics/connecting_routes.csv", names = column_names)

connecting_routes.drop("Unnamed", axis = 1, inplace = True)
connecting_routes.head()

# Loading data into sql database
connecting_routes.to_sql('connecting_routes', con = engine, if_exists = 'replace', chunksize = 1000, index= False)

# Reading data from sql database
sql = 'select * from connecting_routes;'
connecting_routes = pd.read_sql_query(sql, con = engine)

connecting_routes.head()

# connecting_routes = connecting_routes.iloc[0:150, 1:8]
connecting_routes.columns

# Check for missing values in all columns
missing_values = connecting_routes.isnull().sum()

# Print the missing values count for each column
print(missing_values)

# Impute missing values in the "Haults" column with "N"
connecting_routes["Haults"].fillna("N", inplace=True)

# Calculate the mode of the "Machinery" 
mode_value = connecting_routes["Machinery"].mode()[0]
# Impute missing values in the "Machinery" column with the mode 
connecting_routes["Machinery"].fillna(mode_value, inplace = True)

print(missing_values)

for_g = nx.DiGraph()
for_g = nx.from_pandas_edgelist(connecting_routes, source = 'Main Airport', 
                                target = 'Destination Airport')


print(nx.info(for_g))

# #  centrality:-
# 
# 
# **Degree centrality** is defined as the number of links incident upon a node (i.e., the number of ties that a node has). ... Indegree is a count of the number of ties directed to the node (head endpoints) and outdegree is the number of ties that the node directs to others (tail endpoints).
# 
# **Eigenvector Centrality** The adjacency matrix allows the connectivity of a node to be expressed in matrix form. So, for non-directed networks, the matrix is symmetric.Eigenvector centrality uses this matrix to compute its largest, most unique eigenvalues.
# 
# **Closeness Centrality** An interpretation of this metric, Centralness.
# 
# **Betweenness centrality** This metric revolves around the idea of counting the number of times a node acts as a bridge.


data = pd.DataFrame({"closeness":pd.Series(nx.closeness_centrality(for_g)),
                     "Degree": pd.Series(nx.degree_centrality(for_g)),
                     "eigenvector": pd.Series(nx.eigenvector_centrality(for_g)),
                     "betweenness": pd.Series(nx.betweenness_centrality(for_g))}) 

data


# Visual Representation of the Network
for_g = nx.DiGraph()
for_g = nx.from_pandas_edgelist(connecting_routes, source = 'Main Airport', target = 'Destination Airport')

# Create a figure and axis
f, ax = plt.subplots(figsize=(10, 10))

# Define the layout for the graph (you can adjust the parameters)
pos = nx.spring_layout(for_g, k=0.015)

# Draw the networkx graph with customized node size and color
nx.draw_networkx(for_g, pos, ax=ax, node_size=50, node_color='red', with_labels=False)

# Set axis properties (you can customize further)
ax.set_title("NetworkX Graph - Connecting Routes", fontsize=16)
ax.set_xticks([])
ax.set_yticks([])

# Display the graph
plt.show()

f.savefig("graph_connecting_routes.png")

os.getcwd()

# Define column names as a list
Flight_hault_cols = ("ID","Name","City","Country","IATA_FAA","ICAO","Latitude","Longitude","Altitude","Time","DST","Tz database time")

# Reading data from local drive
Flight_hault = pd.read_csv(r"C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/8. Network Analytics/flight_hault.csv", names = Flight_hault_cols)

Flight_hault.head()

# Check for missing values in all columns
missing_values = Flight_hault.isnull().sum()
# Print the missing values count for each column
print(missing_values)

# ICAO and IATA codes serve as identifiers for airports

# Calculate the mode of the "IATA_FAA" 
mode_code = Flight_hault['IATA_FAA'].mode()[0]
# Function to replace numeric characters with the mode value
def replace_numeric_with_mode(code):
    return ''.join([mode_code if c.isdigit() else c for c in code])

# Apply the function to each airport code in the dataset
Flight_hault['IATA_FAA'] =  Flight_hault['IATA_FAA'].apply(replace_numeric_with_mode)

# Impute missing values in the "Machinery" column with the mode 
Flight_hault["IATA_FAA"].fillna(mode_code, inplace = True)

# Calculate the mode of the "ICAO" 
mode_val = Flight_hault["ICAO"].mode()[0]
# Function to replace numeric characters with the mode value
def replace_numbers_value(val):
    return ''.join([mode_val if v.isdigit() else v for v in val])

# Apply the function to each airport code in the dataset
Flight_hault["ICAO"] =  Flight_hault["ICAO"].apply(replace_numbers_value)
# Impute missing values in the "ICAO" column with the mode 
Flight_hault["ICAO"] = Flight_hault["ICAO"].replace("\\N", mode_val)
# Impute missing values in the "Machinery" column with the mode 
Flight_hault["ICAO"].fillna(mode_val, inplace = True)

# Check the type of your data
data_type = type(Flight_hault)

# Print the result
print(data_type)

hault = nx.DiGraph()
hault = nx.from_pandas_edgelist(Flight_hault, source = 'IATA_FAA', 
                                target = 'ICAO')


print(nx.info(hault))

df = pd.DataFrame({"closeness":pd.Series(nx.closeness_centrality(hault)),
                     "Degree": pd.Series(nx.degree_centrality(hault)),
                     "eigenvector": pd.Series(nx.eigenvector_centrality(hault)),
                     "betweenness": pd.Series(nx.betweenness_centrality(hault))}) 

df

# Create a figure and axis
f, ax = plt.subplots(figsize=(10, 10))
# Define the layout for the graph (you can adjust the parameters)
pos = nx.spring_layout(hault, k=0.015)
# Draw the networkx graph with customized node size and color
nx.draw_networkx(hault, pos, ax=ax, node_size=30, node_color='green', with_labels=False)

# Set axis properties (you can customize further)
ax.set_title("NetworkX Graph - Flight Hault", fontsize=16)
ax.set_xticks([])
ax.set_yticks([])

# Display the graph
plt.show()

f.savefig("graph_filght_hault.png")

os.getcwd()