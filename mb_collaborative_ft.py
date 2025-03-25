import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import cross_validate, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
ratings_explicit = pd.read_csv("Ratings.csv")
books = pd.read_csv("Books.csv", dtype={'ISBN': str}, low_memory=False)

# Rename columns for consistency
ratings_explicit.rename(columns={'User-ID': 'user_id', 'ISBN': 'isbn', 'Book-Rating': 'book_rating'}, inplace=True)

# Filter users with at least 3 ratings
user_ratings_threshold = 3
filter_users = ratings_explicit['user_id'].value_counts()
filter_users_list = filter_users[filter_users >= user_ratings_threshold].index.to_list()
df_ratings_top = ratings_explicit[ratings_explicit['user_id'].isin(filter_users_list)]

# Filter top 10% most rated books
book_ratings_threshold_perc = 0.1
book_ratings_threshold = int(len(df_ratings_top['isbn'].unique()) * book_ratings_threshold_perc)
filter_books_list = df_ratings_top['isbn'].value_counts().head(book_ratings_threshold).index.to_list()
df_ratings_top = df_ratings_top[df_ratings_top['isbn'].isin(filter_books_list)]

# Check remaining books and users
print(f"Remaining books after filtering: {len(df_ratings_top['isbn'].unique())}")
print(f"Remaining users after filtering: {len(df_ratings_top['user_id'].unique())}")

# Load data into Surprise dataset format
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df_ratings_top[['user_id', 'isbn', 'book_rating']], reader)

# Train and evaluate SVD model
model_svd = SVD()
cv_results_svd = cross_validate(model_svd, data, cv=3)
print("SVD Performance:\n", pd.DataFrame(cv_results_svd).mean())

# Train the best SVD model
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD(n_factors=80, n_epochs=100, lr_all=0.005, reg_all=0.2)
model.fit(trainset)

# Make predictions
predictions = model.test(testset)

# Convert predictions to DataFrame
df_pred = pd.DataFrame(predictions, columns=['user_id', 'isbn', 'actual_rating', 'pred_rating', 'details'])
df_pred['abs_err'] = abs(df_pred['pred_rating'] - df_pred['actual_rating'])
df_pred.drop(['details'], axis=1, inplace=True)

# Round predicted ratings for visualization
df_pred['pred_rating_round'] = df_pred['pred_rating'].round()

# Plot rating distributions
palette = sns.color_palette("RdBu", 11)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))

sns.countplot(x='actual_rating', data=df_pred, hue='actual_rating', palette=palette, legend=False, ax=ax1)
ax1.set_title('Distribution of actual ratings of books in the test set')

sns.countplot(x='pred_rating_round', data=df_pred, hue='actual_rating', palette=palette, legend=False, ax=ax2)
ax2.set_title('Distribution of predicted ratings of books in the test set')

plt.show()

# Mean absolute error for each rating
df_pred_err = df_pred.groupby('actual_rating')['abs_err'].mean().reset_index()

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))

sns.histplot(df_pred['abs_err'], bins=20, kde=True, color='#2f6194', ax=ax1)
ax1.set_title('Distribution of absolute error in test set')

sns.barplot(x='actual_rating', y='abs_err', data=df_pred_err, hue='actual_rating', palette=palette, legend=False, ax=ax2)
ax2.set_title('Mean absolute error for rating in test set')

plt.show()

# Merge with book details
df_books = books[['ISBN', 'Book-Title']].copy()
df_books.rename(columns={'ISBN': 'isbn', 'Book-Title': 'book_title'}, inplace=True)

df_ext = df_ratings_top.merge(df_books, on='isbn', how='left')
df_ext = df_ext.merge(df_pred[['isbn', 'user_id', 'pred_rating']], on=['isbn', 'user_id'], how='left')

# Check NaN values in predictions
print(f"Number of NaN values in pred_rating: {df_ext['pred_rating'].isna().sum()}")

# Select a specific user and recommend books
selected_user_id = 193458
df_user = df_ext[df_ext['user_id'] == selected_user_id]

# Get books that the user rated highly but don't have predictions
df_highly_rated_missing = df_user[df_user['pred_rating'].isna() & (df_user['book_rating'] >= 9)]
print("\nBooks rated highly by the user but not in predictions:")
if not df_highly_rated_missing.empty:
    print(df_highly_rated_missing.sample(min(5, len(df_highly_rated_missing))))
else:
    print("No books meet the criteria.")

# Recommend top 5 books based on predicted rating
df_user_pred = df_user[df_user['pred_rating'].notna()].sort_values('pred_rating', ascending=False)
print("\nTop 5 book recommendations for the user:")
if not df_user_pred.empty:
    print(df_user_pred.head(5))
else:
    print("No recommendations available.")

# Display books with the highest actual ratings from the user
df_user_actual = df_user[df_user['pred_rating'].notna()].sort_values('book_rating', ascending=False)
print("\nBooks with the highest actual ratings from the user:")
if not df_user_actual.empty:
    print(df_user_actual.head(5))
else:
    print("No books found with actual ratings.")
