"""

# Data Mining Unsupervised Learning / Descriptive Modeling - Association Rule Mining

# Problem Statement

# Sales of the bookstore has been less due to the online selling of books and widespread Internet access.
Store owner realised this because their annual growth started to collapse

# Heritage bookstore wants to gain its popularity back and increase the footfall of customers 
The customers purchasing habits needed to be understood by finding the association between the products in the customers transactions. 
This information can help bookstore to identify the ways to improve sales and by devising strategies to increase revenues and develop effective sales strategies.

# `CRISP-ML(Q)` process model describes six phases:
# 
# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Model Evaluation
# 5. Deployment
# 6. Monitoring and Maintenance

#  Business Objective: Revive Popularity & Increase Profits
# 
# Business Constraints: Budget, Reputation

# **Success Criteria**
# 
# - **Business Success Criteria**: Improve the cross selling in Store by value at a 25% improvement of the current rate. 
# 
# - **ML Success Criteria**: Accuracy, Confidence, Patterns 
    Performance : Complete processing within 5 mins on every quarter data
# 
# - **Economic Success Criteria**: Growth Rate, Increase the Book Store profits by 25%

# **Proposed Plan:**
# Identify the Association between the books being purchased by the customers
 from the store
  

# ## Data Collection

# Data: 
#    The daily transactions made by the customers are captured by the store.
# Description:
# A total of 2000 transactions data captured for the month.

# Mlxtend (machine learning extensions) is a Python library of useful tools for the day-to-day data science tasks.

"""

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
import seaborn as sns 

# Suppress the Warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/6. Association_Rules/book.csv")

df.info()

df.isna().sum()

# Calculates the count of each item and displays the most popular books.
count = df.loc[:, :].sum()
count

# Visualizes the popularity of books using a bar plot.
plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('seaborn')
count.sort_values(ascending=False).plot(kind='bar', color = 'violet')
plt.title('Most Popular Books')
plt.xlabel('Items')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Market Basket Analysis
# Identifies items with counts greater than 500 for potential market basket analysis.
popular_items = count[count > 500].index.tolist()  # Items with count greater than 1
print("Popular Items for Market Basket Analysis:", popular_items)

# Customer Segmentation
# Demonstrates customer segmentation and identifies popular items for each segment (for demonstration purposes).
customer_segments = {
    'SegmentA': ['ChildBks', 'YouthBks', 'CookBks'],
    'SegmentB': ['ArtBks', 'GeogBks', 'RefBks', 'DoItYBks'],
    'SegmentC': ['Florence', 'ItalAtlas', 'ItalCook', 'ItalArt']
}

for segment, items in customer_segments.items():
    segment_popular_items = count[items].idxmax()
    print(f"Popular items for {segment}:", segment_popular_items)
print()

# Apply Apriori algorithm
frequent_books = apriori(df, min_support=0.005, max_len = 3, use_colnames=True)
frequent_books

rules = association_rules(frequent_books, metric="lift", min_threshold = 1)
# Display association rules
print("Association Rules:", rules)

# heatmap to visualize the association rules and their metrics.
heatmap_data = rules.pivot(index='antecedents', columns='consequents', values = ['lift', 'confidence', 'support'])
# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap='plasma')
plt.title("Association Rules Heatmap\nLift, Confidence, and Support")
plt.show()

res = rules.sort_values('support', ascending = False).head(7)

# Data Visualization
# rules data for top 7 frequent books 
f = [
    {'antecedents': "'CookBks'", 'consequents': "'ChildBks'", 'confidence': 0.59, 'lift': 1.4},
    {'antecedents': "'ChildBks'", 'consequents': "'CookBks'", 'confidence': 0.6, 'lift': 1.4},
    {'antecedents': "'GeogBks'", 'consequents': "'ChildBks'", 'confidence': 0.7, 'lift': 1.67},
    {'antecedents': "'ChildBks'", 'consequents': "'GeogBks'", 'confidence': 0.46, 'lift': 1.67},
    {'antecedents': "'CookBks'", 'consequents': "'GeogBks'", 'confidence': 0.44, 'lift': 1.62},
    {'antecedents': "'GeogBks'", 'consequents': "'CookBks'", 'confidence': 0.69, 'lift': 1.62},
    {'antecedents': "'DoItYBks'", 'consequents': "'CookBks'", 'confidence': 0.66, 'lift': 1.54}
]

# Extract confidence and lift values for visualization
confidences = [rule['confidence'] for rule in f]
lifts = [rule['lift'] for rule in f]

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(confidences, lifts, color='green', edgecolors='black', alpha=0.7)
plt.title("Association Rules Visualization")
plt.xlabel("Confidence")
plt.ylabel("Lift")

# Add gridlines
plt.grid(True)

# Show the plot
plt.show()

# Removing frozenset from dataframe
rules['antecedents'] = rules['antecedents'].astype('string')
rules['consequents'] = rules['consequents'].astype('string')

rules['antecedents'] = rules['antecedents'].str.removeprefix("frozenset({")
rules['antecedents'] = rules['antecedents'].str.removesuffix("})")

rules['consequents'] = rules['consequents'].str.removeprefix("frozenset({")
rules['consequents'] = rules['consequents'].str.removesuffix("})")

# here's a focused summary of the top 7 association rules
# offering discounts when customers buy "CookBks" and "ChildBks" together could lead to increased sales in both categories.
# Customers who purchase "CookBks" are 1.40 times more likely to also purchase "ChildBks," with a confidence of 59.40%.
# Customers who purchase "ChildBks" are 1.40 times more likely to also purchase "CookBks," with a confidence of 60.52%.
# Customers who purchase "GeogBks" are 1.67 times more likely to also purchase "ChildBks," with a high confidence of 70.65%.
# Customers who purchase "GeogBks" are 1.62 times more likely to also purchase "CookBks," with a high confidence of 69.75%.
# Customers who purchase "GeogBks" are 1.62 times more likely to also purchase "CookBks," with a high confidence of 69.75%
# Customers who purchase "DoItYBks" are 1.54 times more likely to also purchase "CookBks," with a confidence of 66.49%.
# Customers who purchase "ChildBks" are 1.67 times more likely to also purchase "GeogBks," with a confidence of 46.10%.