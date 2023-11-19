'''
There are three datasets given (Facebook, Instagram, and LinkedIn). Construct and visualize the following networks:
●	circular network for Facebook
●	star network for Instagram
●	star network for LinkedIn

Create a network using an adjacency matrix (undirected only). 

'''

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

''' Visualize circular network for Facebook '''
fb = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/8. Network Analytics/facebook.csv")
# fb - binary adjacency matrix where rows and columns correspond to nodes, and the values indicate the presence (1) or absence (0) of edges (connections) between nodes.

fb.head()

# Convert the DataFrame to a NumPy array
adjacency_matrix = fb.values

# Create circular network for Facebook
fb_graph = nx.Graph(adjacency_matrix)

# Define node positions for the circular layout
pos = nx.circular_layout(fb_graph)

# Create a new figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Draw the circular network
nx.draw(fb_graph, pos, with_labels=True, ax=ax, node_size=300, node_color='skyblue', font_size=10, font_color='black', font_weight='bold')

# Show the plot
plt.show()

''' Visualize star network for Instagram '''
insta = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/8. Network Analytics/instagram.csv")

insta.head()

# Convert the DataFrame to a NumPy array
adj_matrix = insta.values

# Create circular network for Facebook
insta_graph = nx.Graph(adj_matrix)

# Define node positions for the circular layout
pos = nx.spring_layout(insta_graph)

# Create a new figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Draw the circular network
nx.draw(insta_graph, pos, with_labels=True, ax=ax, node_size=300, node_color='red', font_size=10, font_color='black', font_weight='bold')

# Show the plot
plt.show()


''''Visualize star network for LinkedIn '''
ldn = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/8. Network Analytics/linkedin.csv")


ldn.head()

# Convert the DataFrame to a NumPy array
adcy_matrix = ldn.values

# Create circular network for Facebook
ldn_graph = nx.Graph(adcy_matrix)

# Define node positions for the circular layout
pos = nx.spring_layout(ldn_graph)

# Create a new figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Draw the circular network
nx.draw(ldn_graph, pos, with_labels=True, ax=ax, node_size=300, node_color='yellow', font_size=10, font_color='black', font_weight='bold')

# Show the plot
plt.show()