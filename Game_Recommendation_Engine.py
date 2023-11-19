'''
CRISP-ML(Q) process model describes six phases:
# - Business and Data Understanding
# - Data Preparation
# - Model Building
# - Model Evaluation and Hyperparameter Tuning
# - Model Deployment
# - Monitoring and Maintenance

# Business Problem: Video gaming industry conducted a survey to build a 
recommendation engine so that the store can improve the sales of its gaming DVDs.

# Objective(s): Maximize DVD Sales
# Constraint(s): Optimize DVD sales growth while ensuring profitability.

Success Criteria:
    a. Business: Increase the sales by 18% to 20%
    b. ML: Develop a recommendation engine
    c. Economic: Additional revenue of $300K to $500K
    
    Data Collection: 
        Dimension: 5000 rows and 3 columns
        1. userId : Identifies the user who interacted with the game.
        2. game : Represents the title of the video game.
        3. rating : Indicates the user's rating or feedback for the game, reflecting their satisfaction or preference.
'''

# Importing all required libraries, modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import cosine_similarity

# import Dataset 
df = pd.read_csv(r"C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/7. Recommendation Engine/game.csv", encoding = 'utf8')

# Basic statistics
num_users = df['userId'].nunique()
num_games = df['game'].nunique()
avg_rating = df['rating'].mean()
min_rating = df['rating'].min()
max_rating = df['rating'].max()

# Distribution of ratings
plt.figure(figsize=(8, 5))
sns.histplot(df['rating'], bins=5, kde=True)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Visualize user activity (number of ratings per user)
user_activity = df.groupby('userId')['rating'].count().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.histplot(user_activity, bins=20, kde=True)
plt.title('User Activity (Number of Ratings per User)')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.show()

# Visualize game popularity (number of ratings per game)
game_popularity = df.groupby('game')['rating'].count().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.histplot(game_popularity, bins=20, kde=True)
plt.title('Game Popularity (Number of Ratings per Game)')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Games')
plt.show()

# Check for Missing Values in all columns: 
missing_values_count = df[['userId', 'game', 'rating']].isnull().sum()

# Print the missing values count for each column
print("Missing Values Count for Each Column:")
print(missing_values_count)

df.head()
df.groupby('game')['rating'].mean().sort_values(ascending = False).head()
df.groupby('game')['rating'].count().sort_values(ascending = False).head()

ratings = pd.DataFrame(df.groupby('game')['rating'].mean())
ratings.head()

ratings['number of ratings'] = pd.DataFrame(df.groupby('game')['rating'].count())

ratings.sort_values('number of ratings', ascending = False).head(10)


plt.figure(figsize = (10,4))
ratings['number of ratings'].hist(bins = 20)

plt.figure(figsize = (10,4))
ratings['rating'].hist(bins = 40)

sns.jointplot(x = 'rating', y = 'number of ratings', data = ratings, alpha = 0.8)

# Now let's create a user-item matrix that has the user id's on one axis and game title on another axis. 
# Each cell will then consist of the rating the user gave to that game. 
# Note : There will be a lot of NaN values, because most of the people have not played most of thr games.
 
# Create a user-item matrix
user_game_matrix = df.pivot_table(index = 'userId', columns = 'game', values = 'rating')
user_game_matrix.head()

# Fill missing values with zeros
user_game_matrix = user_game_matrix.fillna(0)

# Replace any remaining NaN values with zeros
user_game_matrix = user_game_matrix.replace(np.nan, 0)
user_game_matrix.head()
# Initialize an empty DataFrame to store top correlated games
final_top_similar_games = pd.DataFrame(columns=['User_ID', 'Game', 'Correlation'])

# Loop through each user ID to calculate and append top similar games
for target_user_id in df['userId'].unique():
    # Filter the user-game matrix for the current user
    user_item_matrix = user_game_matrix.loc[target_user_id]
    
    # Calculate cosine similarity for all games for the current user
    similarities = cosine_similarity(user_game_matrix, [user_item_matrix])
    similarities = similarities.reshape(-1)  # Flatten the 2D array
    
    # Sort games by cosine similarity
    similar_games = user_game_matrix.columns[similarities.argsort()[::-1]]
    
    # Create a DataFrame with the top correlated games for the current user
    top_correlations_df = pd.DataFrame({
        'User_ID': target_user_id,
        'Game': similar_games[1:11],  # Exclude the current user's game
        'Correlation': similarities[similarities.argsort()[::-1]][1:11]  # Exclude the current user's game
    })
    
    # Append the top correlated games for the current user to the final DataFrame
    final_top_similar_games = pd.concat([final_top_similar_games, top_correlations_df], ignore_index=True)

# Handle possible division by zero or degrees of freedom issues
final_top_similar_games.fillna(0, inplace=True)

# Display the final top correlated games for all users
print(final_top_similar_games)

# The code provide a more efficient and comprehensive list of top similar games in the given dataset.