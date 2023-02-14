# %% IMPORTS & INIT
import pandas as pd
import numpy as np
import os


# %% DATA IMPORTS
users = pd.read_csv(os.path.join("Data_Final", "user_details.csv"))
users_books = pd.read_csv(os.path.join("Data_Final", "bookshelves.csv"))
books = pd.read_csv(os.path.join("Data_Final", "all_books.csv"))


# %%
