# %%
import plotly.express as px
from yellowbrick.cluster.icdm import InterclusterDistance
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from pathlib import Path
import os

from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

dir = os.getcwd()
os.chdir(dir)


# %% ### 1) Data Transformation

# join the books and users information to get book counts
# df_users = pd.read_csv("../Data/Data_Final/user_details.csv")
# df_book_users = pd.read_csv("../Data/Data_Final/bookshelves.csv")
# df_books = pd.read_csv("../Data/Data_Final/all_books.csv")

# df_books_tmp = pd.merge(df_books, df_book_users, left_on="book_id", right_on="BOOKID", how="inner")  # no missing books to users
# df_tmp = pd.merge(df_books_tmp, df_users, on="USERID", how="inner")  # some books have not been read by users

# book_counts = df_tmp.groupby("BOOKID").size()
# book_counts.name = "Book_Counts"

# df = pd.merge(df_books, book_counts, left_on="book_id", right_index=True, how="left")
df = pd.read_csv("../Data/Data_Final/all_books.csv")
df['isbn'] = df['isbn'].astype(str)

# Description
# do some preliminary data exploration

# %%
df.info()

# %%
df.describe()

# %%
df.isnull().sum()

# %%
(df.isnull().sum() / df.shape[0]) * 100

# %%
df.corr()

# %%
scatter_matrix(df)

# %% ### 2a) Data Transformation

# remove unique identifiers/unnecessary column
df = df.dropna(how="any")
df_store = df[["isbn", "book_id", "genre", "author", "title"]] # store these for recommendation system
df = df.drop(columns=["url", "isbn", "book_id", "title"])

# split punlish into year month day columns from string
df_date = df['publish'].str.replace("First published ", "").str.replace("Published ", "").str.replace("Expected ", "").str.replace("publication ", "").str.split(" ", expand=True)

# Get Dates
df_date.columns = ["Month", "Day", "Year"]
df["year"] = df_date["Year"].astype(int)

df = df.drop(columns="publish")

# Get Genre Counts Frequency Encoded
df_genre_counts = df["genre"].value_counts()
df['genre_freq'] = df['genre'].map(df_genre_counts) / df.shape[0]
df = df.drop(columns="genre")

# Get Author Counts Frequency Encoded
df_author_counts = df["author"].value_counts()
df['author_freq'] = df['author'].map(df_author_counts) / df.shape[0]
df = df.drop(columns="author")

# get number of pages and add a field if the value is missing or not
df_books = df["book_pages"].str.replace(",", "").str.split(" ", expand=True)

non_nums = []
for item in df_books[0].unique():
    try:
        int(item)
    except:
        non_nums.append(item)

df["pages"] = df_books[0].replace(non_nums, np.nan).astype(float)
df["page_missing"] = [1 if val is np.nan else 0 for val in df["pages"]]

df = df.drop(columns="book_pages")
plt.hist(np.log(df["pages"]))

# create pipeline to do imputation using median
# one hot encoding of categorial variables
# and power transformation
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])
min_max_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", MinMaxScaler()),
])
one_cat_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("one_hot", OneHotEncoder(drop='first')),
])
pow_num_pipeline = Pipeline([
    ("impute", IterativeImputer(initial_strategy="median")),
    ("log", PowerTransformer()),
    ("standardize", StandardScaler()),
])

# Transformation
num_attrs = ['rating']
min_max_attrs = ['genre_freq', 'author_freq', 'year']
one_cat_attrs = ['award', 'page_missing']
pow_num_attrs = ['num_of_rating', 'num_of_review', 'pages']

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attrs),
    ("min_max", min_max_pipeline, min_max_attrs),
    ("one_cat", one_cat_pipeline, one_cat_attrs),
    ("pow_num", pow_num_pipeline, pow_num_attrs)
], sparse_threshold=0)

# run pipeline on training and testing data
df_prepared = preprocessing.fit_transform(df)

df_prep_out = pd.DataFrame(
    df_prepared,
    columns=preprocessing.get_feature_names_out(),
    index=df.index)

df_prep_out.head()
df_prep_out.columns = ['rating', 'genre_freq', 'author_freq', 'year', 'award', 'num_ratings', 'num_reviews', 'num_pages']
df_prep_out.corr()


# %% ### 3) Model Testing

# visualize the KMeans data to determine number of cluster centers
km = KMeans(n_init=10, random_state=42)
vis = KElbowVisualizer(km, k=(2, 30))

vis.fit(df_prep_out)
vis.show()

# plot the first two principal components
km = KMeans(n_clusters=9, n_init=10, random_state=42)
vis_clust = InterclusterDistance(km)
vis_clust.fit(df_prep_out)
vis_clust.show()

# visualize the principal components in 2D and 3D
pca = PCA(n_components=df_prep_out.shape[1]-1)
X_PCA = pca.fit_transform(df_prep_out)
df_PCA = pd.DataFrame(X_PCA, columns=[f"PC{i}" for i in range(1, len(df_prep_out.columns))])

# calculate the components
df_components = pd.DataFrame(pca.components_, columns=df_prep_out.columns)
df_components = df_components.T
df_components.columns = [f"PC{i}" for i in range(1, len(df_prep_out.columns))]

# check metrics
pca.explained_variance_ratio_.round(2)
df_PCA.head()
df_components.round(2)

# transform data by PCA
fitted = km.fit(df_prep_out)
labels = fitted.labels_
centers = fitted.cluster_centers_

# get centers of clusters in a dataframe for plotting
pca_centers = pca.transform(centers)
df_centers = pd.DataFrame(pca_centers, columns=[f"PC{i}" for i in range(1, len(df_prep_out.columns))])
df_centers

# plot PC1 vs PC2; update pc1 and pc2 to see whichever PCs you want
pc1 = 1
pc2 = 2
pcs = [f"PC{i}" for i in range(1, len(df_prep_out.columns))]
plt.style.use('ggplot')
plt.scatter(df_PCA[pcs[pc1-1]], df_PCA[pcs[pc2-1]], c=labels, cmap='rainbow', s=2)
plt.scatter(df_centers[pcs[pc1-1]], df_centers[pcs[pc2-1]], c='black', s=5)
plt.title(f"{pcs[pc1-1]} by {pcs[pc2-1]}")
plt.xlim((max(df_PCA[pcs[pc1-1]].min(), -10) - 0.2, min(df_PCA[pcs[pc1-1]].max(), 10) + 0.2))
plt.ylim((max(df_PCA[pcs[pc2-1]].min(), -10) - 0.2, min(df_PCA[pcs[pc2-1]].max(), 10) + 0.2))

# plot 3d PC1 vs PC2 vs PC3; update pc1, pc2, and pc3 to see whichever PCs you want
pc1 = 1
pc2 = 2
pc3 = 3
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df_PCA[pcs[pc1-1]], df_PCA[pcs[pc2-1]], df_PCA[pcs[pc3-1]], c=labels, cmap='rainbow', s=2)
ax.scatter(df_centers[pcs[pc1-1]], df_centers[pcs[pc2-1]], df_centers[pcs[pc3-1]], c='black', s=9)
ax.set_xlim((-6, 8))
ax.set_ylim((-8, 4))
ax.set_zlim((-4, 8))
ax.set_xlabel(f"PC{pc1}")
ax.set_ylabel(f"PC{pc2}")
ax.set_zlabel(f"PC{pc3}")

plt.show()

# Visualize the data in 3d using plotly for interative plots
pc1 = 1
pc2 = 2
pc3 = 3
df = pd.concat([df_PCA, pd.Series(labels, name="Labels")], axis=1)
df.head()
fig = px.scatter_3d(df, x=f"PC{pc1}", y=f"PC{pc2}", z=f"PC{pc3}",
                    color='Labels', range_x=[-6, 8], range_y=[-8, 4], range_z=[-4, 8])
fig.show(renderer="browser")


# %% ### 4) For each user get top clusters to recommend

# determine the books read by each user and raw book counts
df_users = pd.read_csv("../Data/Data_Final/user_details.csv")
df_book_users = pd.read_csv("../Data/Data_Final/bookshelves.csv")
labels = fitted.labels_

# get raw book counts for the data
df_books = pd.concat([df_prep_out, df_store["book_id"], pd.Series(labels, name="Label")], axis=1)
df_books = pd.merge(df_books, book_counts, left_on='book_id', right_index=True)
df_books['Book_Counts'] = df_books['Book_Counts'].replace({np.nan: 0})
df_books['Book_Counts'] = np.square(df_books['Book_Counts']) # gives more probability to more read books
df_books['Book_Counts'] = df_books['Book_Counts'] + 1 # add 1 so that when doing probabilities there is a chance a new book is chosen

# determine the books a user has read
df_books_tmp = pd.merge(df_books, df_book_users, left_on="book_id", right_on="BOOKID", how="inner")  # no missing books to users
df_all = pd.merge(df_books_tmp, df_users, on="USERID", how="left")  # some books have not been read by users
user_list = df_all["USERID"].unique().tolist()
df_all = df_all.drop(columns=["BOOKID", "age", "gender", "Country", "Book_Counts"])


# %% 
# subset the data for a single user to see the recommendations
user = user_list[0]
count = 0
for user in user_list:
    df_test = df_all.loc[df_all['USERID'] == user, :]
    # read_book_ids = df_test['book_id'].to_list()
    # df_test = df_test.drop(columns=["USERID", "Label", "book_id"])
    X_train, X_test = train_test_split(df_test, test_size = 0.3, random_state=42)
    read_book_ids = X_train['book_id'].to_list()
    test_read_book_ids = X_test['book_id'].to_list()
    X_train = X_train.drop(columns=["USERID", "Label", "book_id"])
    X_train = X_train.dropna(how="any")
    X_test = X_test.drop(columns=["USERID", "Label", "book_id"])
    X_test = X_test.dropna(how="any")

    # preds = km.predict(df_test)
    preds = km.predict(X_train)
    test_preds = km.predict(X_test)

    num_topics = {}
    for book in preds:
        if str(book) not in num_topics.keys():
            num_topics[str(book)] = 1
        else:
            num_topics[str(book)] += 1

    total = 0
    for key in num_topics.keys():
        num_topics[key] = round(num_topics[key] / len(preds) * 30)
        total += num_topics[key]

    if total < 30:
        max_key = max(num_topics, key=num_topics.get)
        num_topics[max_key] += 1
    elif total > 30:
        min_key = max(num_topics, key=num_topics.get)
        num_topics[min_key] -= 1

    recommendations = []
    for key in num_topics.keys():
        df_books_sub = df_books.loc[df_books["Label"] == int(key), :]
        df_books_sub = df_books_sub.loc[~(df_books_sub['book_id'].isin(read_book_ids)), :]

        ###################################### finds how many users have read the book and adjusts probability of recommending ########################
        total = 0
        for num in df_books_sub["Book_Counts"]:
            total += num
        df_books_sub['Book_Counts'] = df_books_sub['Book_Counts'] / total

        choices = np.random.choice(df_books_sub['book_id'].to_numpy(), num_topics[key], p=df_books_sub['Book_Counts'].to_numpy(), replace=False)
        recommendations.extend(list(choices))


    idx = df_books[df_books['book_id'].isin(recommendations)].index

    book_recs = df_store.loc[idx, :]
    read_books = df_store.loc[df_store['book_id'].isin(read_book_ids)]
    test_read_books = df_store.loc[df_store['book_id'].isin(test_read_book_ids)]

    read_books
    book_recs
    test_read_books

    both = 0
    overlap_users = []
    for bookA in book_recs['book_id']:
        for bookB in test_read_books['book_id']:
            if bookA == bookB:
                both += 1
                if user not in overlap_users:
                    overlap_users.append(user)

    if both > 0:
        count +=1
        print(f"Num of overlapping books: {both}")

# %% #### Visualize books
# transform predicted books
df_prep_preds = pd.concat([df_prep_out, df_store['book_id']], axis=1)
df_prep_preds = df_prep_preds[df_prep_preds['book_id'].isin(book_recs['book_id'])]
df_prep_preds = df_prep_preds.drop(columns="book_id")
pca_preds = pca.transform(df_prep_preds)
df_pca_preds = pd.DataFrame(pca_preds, columns=[f"PC{i}" for i in range(1, len(df_prep_preds.columns))])

# transform read books
pca_test = pca.transform(df_test)
df_pca_test = pd.DataFrame(pca_test, columns=[f"PC{i}" for i in range(1, len(df_test.columns))])

num_topics

pc1 = 1
pc2 = 2

pcs = [f"PC{i}" for i in range(1, len(df_test.columns))]
plt.style.use('ggplot')
plt.scatter(df_pca_test[pcs[pc1-1]], df_pca_test[pcs[pc2-1]], c='red', s=10)
plt.scatter(df_centers[pcs[pc1-1]], df_centers[pcs[pc2-1]], c='black', s=10)
plt.scatter(df_pca_preds[pcs[pc1-1]], df_pca_preds[pcs[pc2-1]], c='blue', s=10)
plt.title(f"{pcs[pc1-1]} by {pcs[pc2-1]}")
plt.xlim((max(df_pca_test[pcs[pc1-1]].min(), -10) - 0.2, min(df_pca_test[pcs[pc1-1]].max(), 10) + 0.2))
plt.ylim((max(df_pca_test[pcs[pc2-1]].min(), -10) - 0.2, min(df_pca_test[pcs[pc2-1]].max(), 10) + 0.2))

pc1 = 1
pc2 = 2
pc3 = 3

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(df_pca_test[pcs[pc1-1]], df_pca_test[pcs[pc2-1]], df_pca_test[pcs[pc3-1]], c="red", s=10)
ax.scatter(df_centers[pcs[pc1-1]], df_centers[pcs[pc2-1]], df_centers[pcs[pc3-1]], c='black', s=10)
ax.scatter(df_centers[pcs[pc1-1]], df_centers[pcs[pc2-1]], df_centers[pcs[pc3-1]], c='blue', s=10)

ax.set_xlim((-6, 8))
ax.set_ylim((-8, 4))
ax.set_zlim((-4, 8))
ax.set_xlabel(f"PC{pc1}")
ax.set_ylabel(f"PC{pc2}")
ax.set_zlabel(f"PC{pc3}")

plt.show()
