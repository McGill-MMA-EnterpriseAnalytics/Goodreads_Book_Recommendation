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

df = pd.read_csv("../Data/Data_Final/all_books.csv")
df = df.drop(columns = ["Unnamed: 0", "duplicate_books"])
df['family_group'] = df['family_group'].astype(int).astype(str)

# Description
# do some preliminary data exploration

# %%
df.info()

# %%
df.describe()

# %%
# only pages has missing values
(df.isnull().sum() / df.shape[0]) * 100

# %%
# not much correlation in the dataset
df.corr()

# %%
scatter_matrix(df)

# %% ### 2a) Data Transformation

# remove unique identifiers/unnecessary column
df_store = df[["family_group"]] # store these for recommendation system
df = df.drop(columns=["family_group", "title"])

# Get Genre Counts Frequency Encoded
df_genre_counts = df["genre"].value_counts()
df['genre_freq'] = df['genre'].map(df_genre_counts) / df.shape[0]
df = df.drop(columns="genre")

# Get Author Counts Frequency Encoded
df_author_counts = df["author"].value_counts()
df['author_freq'] = df['author'].map(df_author_counts) / df.shape[0]
df = df.drop(columns="author")

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
min_max_attrs = ['genre_freq', 'author_freq']
pow_num_attrs = ['num_of_rating', 'num_of_review', 'pages']

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attrs),
    ("min_max", min_max_pipeline, min_max_attrs),
    ("pow_num", pow_num_pipeline, pow_num_attrs)
], sparse_threshold=0)

# run pipeline on training and testing data
df_prepared = preprocessing.fit_transform(df)

df_prep_out = pd.DataFrame(
    df_prepared,
    columns=preprocessing.get_feature_names_out(),
    index=df.index)

df_prep_out.head()
df_prep_out.columns = ['rating', 'genre_freq', 'author_freq', 'num_ratings', 'num_reviews', 'num_pages']
df_prep_out.corr()


# %% ### 3) Model Testing

# DBSCAN, Gaussian Mixture Models we're tested

# visualize the KMeans data to determine number of cluster centers
# did not use silhouette score due to O(n^2) time complexity
# Used the elbow method to determine number of clusters
km = KMeans(n_init=10, random_state=42)
vis = KElbowVisualizer(km, k=(2, 30))

vis.fit(df_prep_out)
vis.show()

# %%
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
plt.xlabel(pcs[pc1-1])
plt.ylabel(pcs[pc2-1])

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
fig = px.scatter_3d(df.sample(10000), x=f"PC{pc1}", y=f"PC{pc2}", z=f"PC{pc3}",
                    color='Labels', range_x=[-6, 8], range_y=[-8, 4], range_z=[-4, 8])
fig.show(renderer="browser")


# %% ### 4) For each user get top clusters to recommend

# determine the books read by each user and raw book counts
df_users = pd.read_csv("../Data/Data_Final/user_details.csv")
df_users = df_users.drop(columns="Unnamed: 0")
df_book_users = pd.read_csv("../Data/Data_Final/bookshelves 2.csv")
df_book_users = df_book_users.drop(columns="Unnamed: 0")
df_book_users['family_group'] = df_book_users['family_group'].astype(int).astype(str)
labels = fitted.labels_

# %%
# get raw book counts for the data
df_books = pd.concat([df_prep_out, df_store["family_group"], pd.Series(labels, name="Label")], axis=1)
df_books_counts = pd.merge(df_books, df_book_users, on="family_group")
book_counts = df_books_counts.groupby("family_group").size()
book_counts.name = "Book_Counts"
df_books = pd.merge(df_books, book_counts, left_on="family_group", right_index=True)

# %%
df_books['Book_Counts'] = df_books['Book_Counts'].replace({np.nan: 0})
df_books['Book_Counts'] = np.square(df_books['Book_Counts']) # gives more probability to more read books
df_books['Book_Counts'] = df_books['Book_Counts'] + 1 # add 1 so that when doing probabilities there is a chance a new book is chosen

# %%
# determine the books a user has read
df_books_tmp = pd.merge(df_books, df_book_users, on="family_group", how="inner")  # no missing books to users
df_all = pd.merge(df_books_tmp, df_users, on="USERID", how="left")  # some books have not been read by users
user_list = df_all["USERID"].unique().tolist()
df_all = df_all.drop(columns=["age", "gender", "Country"])
df_all['Label'] = df_all["Label"].astype(str)


# %% 
# subset the data for a single user to see the recommendations
# split data into testing and training to see performance
count = 0
i=0
for user in user_list:
    df_test = df_all.loc[df_all['USERID'] == user, :]
    if df_test.shape[0] < 10:
        continue
    X_train, X_test = train_test_split(df_test, test_size = 0.3, random_state=42)
    read_book_ids = X_train['family_group'].to_list()
    test_read_book_ids = X_test['family_group'].to_list()
    X_train = X_train.drop(columns=["USERID", "Label", "family_group", 'Book_Counts'])
    X_train = X_train.dropna(how="any")
    X_test = X_test.drop(columns=["USERID", "Label", "family_group", 'Book_Counts'])
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
        df_books_sub = df_books_sub.loc[~(df_books_sub['family_group'].isin(read_book_ids)), :]

        ###################################### finds how many users have read the book and adjusts probability of recommending ########################
        num_books = 0
        for num in df_books_sub["Book_Counts"]:
            num_books += num
        df_books_sub['Book_Counts'] = df_books_sub['Book_Counts'] / num_books
        
        choices = np.random.choice(df_books_sub['family_group'].to_numpy(), num_topics[key], p=df_books_sub['Book_Counts'].to_numpy(), replace=False)
        recommendations.extend(list(choices))


    idx = df_books[df_books['family_group'].isin(recommendations)].index

    book_recs = df_store.loc[idx, :]
    read_books = df_store.loc[df_store['family_group'].isin(read_book_ids)]
    test_read_books = df_store.loc[df_store['family_group'].isin(test_read_book_ids)]

    both = 0
    overlap_users = []
    for bookA in book_recs['family_group']:
        for bookB in test_read_books['family_group']:
            if bookA == bookB:
                both += 1
                if user not in overlap_users:
                    overlap_users.append(user)

    if both > 0:
        count +=1
        print(f"Num of overlapping books: {both}")

# %%
# 21% of users had at least one book from the testing set that was recommended to them again by the model
print(f"Percentage of Users with at least 1 test book that they have read is {count / len(user_list)}")


# %%
# single user example
user = user_list[2459]
df_test = df_all.loc[df_all['USERID'] == user, :]
read_book_ids = df_test['family_group'].to_list()
df_test = df_test.drop(columns=["USERID", "Label", "family_group", 'Book_Counts'])
df_test = df_test.dropna(how="any")

preds = km.predict(df_test)

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
    df_books_sub = df_books_sub.loc[~(df_books_sub['family_group'].isin(read_book_ids)), :]

    ###################################### finds how many users have read the book and adjusts probability of recommending ########################
    num_books = 0
    for num in df_books_sub["Book_Counts"]:
        num_books += num
    df_books_sub['Book_Counts'] = df_books_sub['Book_Counts'] / num_books
    
    choices = np.random.choice(df_books_sub['family_group'].to_numpy(), num_topics[key], p=df_books_sub['Book_Counts'].to_numpy(), replace=False)
    recommendations.extend(list(choices))


idx = df_books[df_books['family_group'].isin(recommendations)].index

book_recs = df_store.loc[idx, :]
read_books = df_store.loc[df_store['family_group'].isin(read_book_ids)]


# %% #### Visualize books
# transform predicted books
df_prep_preds = pd.concat([df_prep_out, df_store['family_group']], axis=1)
df_prep_preds = df_prep_preds[df_prep_preds['family_group'].isin(book_recs['family_group'])]
df_prep_preds = df_prep_preds.drop(columns="family_group")
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
plt.scatter(df_pca_test[pcs[pc1-1]], df_pca_test[pcs[pc2-1]], c='red', s=10, label="Read Books")
plt.scatter(df_centers[pcs[pc1-1]], df_centers[pcs[pc2-1]], c='black', s=10, label="Centroids")
plt.scatter(df_pca_preds[pcs[pc1-1]], df_pca_preds[pcs[pc2-1]], c='blue', s=10, label="Recommended Books")
plt.title(f"{pcs[pc1-1]} by {pcs[pc2-1]}")
plt.xlim((max(df_pca_test[pcs[pc1-1]].min(), -10) - 0.2, min(df_pca_test[pcs[pc1-1]].max(), 10) + 0.2))
plt.ylim((max(df_pca_test[pcs[pc2-1]].min(), -10) - 0.2, min(df_pca_test[pcs[pc2-1]].max(), 10) + 0.2))
plt.legend()
plt.show()

pc1 = 1
pc2 = 2
pc3 = 3

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(df_pca_test[pcs[pc1-1]], df_pca_test[pcs[pc2-1]], df_pca_test[pcs[pc3-1]], c="red", s=10, label="Read Books")
ax.scatter(df_centers[pcs[pc1-1]], df_centers[pcs[pc2-1]], df_centers[pcs[pc3-1]], c='black', s=10, label="Centroids")
ax.scatter(df_pca_preds[pcs[pc1-1]], df_pca_preds[pcs[pc2-1]], df_pca_preds[pcs[pc3-1]], c='blue', s=10, label="Recommended Books")

ax.set_xlim((max(df_pca_test[pcs[pc1-1]].min(), -10) - 0.2, min(df_pca_test[pcs[pc1-1]].max(), 10) + 0.2))
ax.set_ylim((max(df_pca_test[pcs[pc2-1]].min(), -10) - 0.2, min(df_pca_test[pcs[pc2-1]].max(), 10) + 0.2))
ax.set_zlim((max(df_pca_test[pcs[pc3-1]].min(), -10) - 0.2, min(df_pca_test[pcs[pc3-1]].max(), 10) + 0.2))
ax.set_xlabel(f"PC{pc1}")
ax.set_ylabel(f"PC{pc2}")
ax.set_zlabel(f"PC{pc3}")
ax.legend()

# %%
