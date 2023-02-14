# %% IMPORTS & INIT
import pandas as pd
import numpy as np
import os
data_path = os.path.join(os.getcwd(), 'Data')


# DATA IMPORTS
users = pd.read_csv(os.path.join(data_path, "Data_Intermediary", "user_details.csv"))
users_books = pd.read_csv(os.path.join(data_path, "Data_Intermediary", "bookshelves.csv"))
books = pd.read_csv(os.path.join(data_path, "Data_Intermediary", "all_books.csv"))


# CLEAN THE DATASET
books['book_pages'] = books.book_pages.str.lower()
books['pages'] = np.where(books.book_pages.str.contains('pages'),
                          books.book_pages.str.split(" ").str[0],
                          np.nan)
books['pages'] = books['pages'].astype('Int64')
books['author'] = books.author.str.lower()
books['title'] = books.title.str.lower()
books['genre'] = books.genre.str.lower()


# REMOVE SOME OF THE DUPLICATES (the easy ones bruih)
books['family_group'] = books.groupby(['num_of_rating', 'num_of_review', 'rating', 'title', 'author']).ngroup()
books_counts = books.value_counts('family_group')
books_clean = books.groupby('family_group').agg({'num_of_rating': 'median',
                                                 'num_of_review': 'median',
                                                 'rating': 'median',
                                                 'title': 'first',
                                                 'genre': 'first',
                                                 'author': 'first',
                                                 'pages': 'median', })
books_clean['duplicate_books'] = books_counts
books_clean = books_clean.reset_index()


# CLEAN USER READING HISTORY
user_books_clean = users_books.merge(books[['book_id', 'family_group']],
                                     left_on='BOOKID', right_on='book_id',
                                     how='inner')
user_books_clean = user_books_clean[['USERID', 'family_group']].dropna()


# SAVE DATA
users.to_csv(os.path.join(data_path, 'Data_Final', 'user_details.csv'))
user_books_clean.to_csv(os.path.join(data_path, 'Data_Final', 'bookshelves.csv'))
books_clean.to_csv(os.path.join(data_path, 'Data_Final', 'all_books.csv'))


# %%
