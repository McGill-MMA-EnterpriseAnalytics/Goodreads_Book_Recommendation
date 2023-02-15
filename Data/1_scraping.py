# %% ### IMPORTS
from functions import (async_downloader,
                       get_user_list,
                       get_user_detail)
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
user_time_frame = ['&duration=w', '&duration=m', '&duration=y', '&duration=a']

task_list = []
for comb in itertools.product(*[user_types, user_countries, user_time_frame]):
    task_list.append((comb[0]+comb[1]+comb[2], comb[1].split('=')[1]))
results = []
failed = 0
while True:
    data, failed_list = asyncio.run(async_downloader(get_user_list, task_list))
    results.extend(data)

    if len(failed_list) == failed:
        break
    failed = len(failed_list)
user_list = pd.DataFrame(results).drop_duplicates(0)\
    .rename(columns={0: 'user_id', 1: 'country'})
user_list


# %% ### USER DETAILS
task_list = []
for itm in user_list['user_id'].tolist()[:50]:
    task_list.append((f"https://www.goodreads.com/user/show/{itm}", itm))
results = []
failed = 0
while True:
    data, failed_list = asyncio.run(async_downloader(get_user_detail, task_list))
    results.extend(data)

    if len(failed_list) == failed:
        break
    failed = len(failed_list)
user_details = pd.DataFrame(results)\
    .rename(columns={0: 'user_id', 1: 'age', 2: 'gender'})
user_details


# %% ### USER REVIEW PAGES
task_list = []
for itm in user_list[0].tolist()[:50]:
    task_list.append((f"https://www.goodreads.com/review/list/{}", itm))
results = []
failed = 0
while True:
    data, failed_list = asyncio.run(async_downloader(get_user_detail, task_list))
    results.extend(data)

    if len(failed_list) == failed:
        break
    failed = len(failed_list)


# %% ### USER REVIEW HISTORY
task_list = []
for itm in user_list[0].tolist():
    task_list.append((f"https://www.goodreads.com/user/show/{itm}", itm))
results = []
failed = 0
while True:
    data, failed_list = asyncio.run(async_downloader(get_user_detail, task_list))
    results.extend(data)

    if len(failed_list) == failed:
        break
    failed = len(failed_list)


# %% ### BOOK DATA
task_list = []
for itm in user_list[0].tolist():
    task_list.append((f"https://www.goodreads.com/user/show/{itm}", itm))
results = []
failed = 0
while True:
    data, failed_list = asyncio.run(async_downloader(get_user_detail, task_list))
    results.extend(data)

    if len(failed_list) == failed:
        break
    failed = len(failed_list)
