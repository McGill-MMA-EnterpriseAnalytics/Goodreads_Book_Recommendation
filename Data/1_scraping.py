# %% ### IMPORTS
from functions import (async_downloader,
                       get_user_list,
                       get_user_detail,
                       get_user_review_pages,
                       get_user_read_books,
                       get_book_information)
import pandas as pd
import itertools
import asyncio
import nest_asyncio
nest_asyncio.apply()


# %% ### USERS
user_types = ['https://www.goodreads.com/user/best_reviewers',
              'https://www.goodreads.com/user/top_reviewers',
              'https://www.goodreads.com/user/top_readers']
user_countries = ['?country=CA', '?country=US', '?country=US', '?country=AU',
                  '?country=SC', '?country=NZ', '?country=IE', '?country=GB',
                  '?country=SG', '?country=GB', '?country=DO', '?country=TT',
                  '?country=MT', '?country=BB', '?country=LC', '?country=GY']
user_time_frame = ['&duration=w','&duration=m', '&duration=y', '&duration=a']

task_list = []
for comb in itertools.product(*[user_types, user_countries, user_time_frame]):
    task_list.append((comb[0]+comb[1]+comb[2], comb[1].split('=')[1]))

print("READING USER LIST")
results = []
failed = 0
while True:
    data, task_list = asyncio.run(async_downloader(get_user_list, task_list))
    results.extend(data)
    if len(task_list) == failed or len(task_list) == 0:
        break
    failed = len(task_list)
print()

user_list = pd.DataFrame(results).drop_duplicates(0).rename(columns={0: 'user_id', 1: 'country'})
user_list.to_csv('Data/Data_Intermediary/user_list.csv')


# %% ### USER DETAILS
task_list = []
for itm in user_list['user_id'].tolist():
    task_list.append((f"https://www.goodreads.com/user/show/{itm}", itm))

print("READING USER DETAILS")
results = []
failed = 0
while True:
    data, task_list = asyncio.run(async_downloader(get_user_detail, task_list))
    results.extend(data)
    if len(task_list) == failed or len(task_list) == 0:
        break
    failed = len(task_list)
print()

user_details = pd.DataFrame(results).rename(columns={0: 'user_id', 1: 'age', 2: 'gender'})
user_details.to_csv('Data/Data_Intermediary/user_details.csv')


# %% ### USER REVIEW PAGES
task_list = []
for itm in user_list['user_id'].tolist():
    task_list.append((f"https://www.goodreads.com/review/list/{itm}", itm))

print("READING USER REVIEW PAGES")
results = []
failed = 0
while True:
    data, task_list = asyncio.run(async_downloader(get_user_review_pages, task_list))
    results.extend(data)
    if len(task_list) == failed or len(task_list) == 0:
        break
    failed = len(task_list)
print()

user_review_pages = pd.DataFrame(results).rename(columns={0: 'page_link', 1: 'user_id'})


# %% ### USER READ HISTORY
task_list = []
for itm in user_review_pages.iterrows():
    task_list.append((itm[1][0],itm[1][1]))

print("READING USER READ HISTORY")
results = []
failed = 0
while True:
    data, task_list = asyncio.run(async_downloader(get_user_read_books, task_list))
    results.extend(data)
    if len(task_list) == failed or len(task_list) == 0:
        break
    failed = len(task_list)
print()

user_read_history = pd.DataFrame(results).rename(columns={0: 'book_link', 1: 'user_id'})
user_read_history.to_csv('Data/Data_Intermediary/bookshelves.csv')


# %% ### BOOK DATA
task_list = []
for itm in user_read_history.book_link.drop_duplicates().tolist():
    task_list.append((f"https://www.goodreads.com{itm}",))

print("READING BOOK DATA")
results = []
failed = 0
while True:
    data, task_list = asyncio.run(async_downloader(get_book_information, task_list))
    results.extend(data)
    if len(task_list) == failed or len(task_list) == 0:
        break
    failed = len(task_list)
print()

book_data = pd.DataFrame(results)
book_data.columns = ['url','book_pages','num_of_rating','num_of_review',
                     'genre','publish','author','title','description',
                     'rating','award','isbn']
book_data.to_csv('Data/Data_Intermediary/all_books.csv')


# %%
