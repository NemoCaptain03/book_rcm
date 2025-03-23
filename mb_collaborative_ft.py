import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD, NMF
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV

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

# Load data into Surprise dataset format
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df_ratings_top[['user_id', 'isbn', 'book_rating']], reader)

# Train and evaluate SVD model
model_svd = SVD()
cv_results_svd = cross_validate(model_svd, data, cv=3)
print("SVD Performance:\n", pd.DataFrame(cv_results_svd).mean())

# Train and evaluate NMF model
model_nmf = NMF()
cv_results_nmf = cross_validate(model_nmf, data, cv=3)
print("NMF Performance:\n", pd.DataFrame(cv_results_nmf).mean())

# Hyperparameter tuning using GridSearchCV for SVD
param_grid = {
    'n_factors': [80, 100],
    'n_epochs': [5, 20],
    'lr_all': [0.002, 0.005],
    'reg_all': [0.2, 0.4]
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

# Best RMSE score and parameters
print("Best RMSE score:", gs.best_score['rmse'])
print("Best parameters:", gs.best_params['rmse'])

# Train the best SVD model
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD(n_factors=80, n_epochs=20, lr_all=0.005, reg_all=0.2)
model.fit(trainset)

# Make predictions
predictions = model.test(testset)

# Convert predictions to DataFrame
df_pred = pd.DataFrame(predictions, columns=['user_id', 'isbn', 'actual_rating', 'pred_rating', 'details'])
df_pred['abs_err'] = abs(df_pred['pred_rating'] - df_pred['actual_rating'])
df_pred.drop(['details'], axis=1, inplace=True)

# Merge with book details
df_books = books[['ISBN', 'Book-Title']].copy()
df_books.rename(columns={'ISBN': 'isbn', 'Book-Title': 'book_title'}, inplace=True)

df_ext = df_ratings_top.merge(df_books, on='isbn', how='left')
df_ext = df_ext.merge(df_pred[['isbn', 'user_id', 'pred_rating']], on=['isbn', 'user_id'], how='left')

# Select a specific user and recommend books
selected_user_id = 193458
df_user = df_ext[df_ext['user_id'] == selected_user_id]

# Get books that the user rated highly but don't have predictions
print("\nBooks rated highly by the user but not in predictions:")
print(df_user[df_user['pred_rating'].isna() & (df_user['book_rating'] >= 9)].sample(5))

# Recommend top 5 books based on predicted rating
print("\nTop 5 book recommendations for the user:")
print(df_user[df_user['pred_rating'].notna()].sort_values('pred_rating', ascending=False).head(5))

# Display books with the highest actual ratings from the user
print("\nBooks with the highest actual ratings from the user:")
print(df_user[df_user['pred_rating'].notna()].sort_values('book_rating', ascending=False).head(5))
