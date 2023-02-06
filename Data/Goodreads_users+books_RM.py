#%%
# 
# Asynchronous
# import nest_asyncio
# nest_asyncio.apply()
import asyncio
import aiohttp
import json

import tqdm
import requests
from pprint import pprint
import datetime
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np
import json

import re
import os

cookies = {'ccsid': '027-9406763-1350835',
    '__qca': 'P0-867993267-1671835417644',
    'p': 'BFnM5Q76wF0shLrczPEWD6Ai6glvalBOI0HQnM7wAheYffUa',
    'likely_has_account': 'true',
    'srb_8': '0_ar',
    'locale': 'en',
    'csm-sid': '438-4025740-7454350',
    'allow_behavioral_targeting': 'true',
    'session-id': '147-4067731-5921029',
    'lc-main': 'en_US',
    'logged_out_browsing_page_count': '2',
    'ubid-main': '131-6414756-0108314',
    'csm-hit': 'tb:9FKSJEW3PZPAQZKSV74Z+s-9FKSJEW3PZPAQZKSV74Z|1675549894918&t:1675549894919&adb:adblk_yes',
    'session-id-time': '2306269930l',
    'session-token': 'fmlMtndkzGiWYItedsruMqFs9+7dbrKkjWxwS050IWs57BCY8RQJGXxU8qPn+6lNucog+VtoV/qrLn6eBpe8GwFqeSA75LzytrwlvgKScLWr62XYbJJg36UcoVnTCNaBAUkT+jhZdWNLpHXEQ/T3fsq03ctsMB7GChLcVrF/10rEk0ETQ5LmuUN5ordUvBjgbgf/wLazQPbG5Ia8rmhh/Gyptwz9N0ilyQSLx5h7pVc1fl4z+U3O3seFJdO2xI8M',
    'x-main': '"QKUGsTWzgOIn1MQ?MLW6M7?W9sXe6vT8lOfYIRdLo30O9fDyRRuxqM6Up6QliSso"',
    'at-main': 'Atza|IwEBIE9-j0Q56LUPug78YuRWyKfqUI-71kqBUKBSv9o0pgXoEdWRTp3QanY2PVmzS5kuRKhJ-2Qf2Y2Xza8aOzNTehTteyzG_d1HL64CDYGBvcFiM3J3K6q-_iEKTHPqIuFGTC7GBQffrkFancRYWCUTNK7ii9qpL1Qvc-lEAmj5xJFsos2I98msXxsTvgVFMUfr1rhJK6CfvkpCAIziz2lPcltlQCWaFsR10_Le_ytyKVYeIZ18m-F1bXhhejn7mTwQJl0',
    'sess-at-main': '"f/c/JR/xicYEmsJ7sGnJySc4uAQ6ojXuaOFFChXMGho="',
    '_session_id2': '64c06a0ae25fee31093af8a3bf3f79e1',}
headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,/;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Language': 'en-US,en;q=0.9',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    # 'Cookie': 'ccsid=027-9406763-1350835; __qca=P0-867993267-1671835417644; p=BFnM5Q76wF0shLrczPEWD6Ai6glvalBOI0HQnM7wAheYffUa; likely_has_account=true; srb_8=0_ar; locale=en; csm-sid=438-4025740-7454350; allow_behavioral_targeting=true; session-id=147-4067731-5921029; lc-main=en_US; logged_out_browsing_page_count=2; ubid-main=131-6414756-0108314; csm-hit=tb:9FKSJEW3PZPAQZKSV74Z+s-9FKSJEW3PZPAQZKSV74Z|1675549894918&t:1675549894919&adb:adblk_yes; session-id-time=2306269930l; session-token=fmlMtndkzGiWYItedsruMqFs9+7dbrKkjWxwS050IWs57BCY8RQJGXxU8qPn+6lNucog+VtoV/qrLn6eBpe8GwFqeSA75LzytrwlvgKScLWr62XYbJJg36UcoVnTCNaBAUkT+jhZdWNLpHXEQ/T3fsq03ctsMB7GChLcVrF/10rEk0ETQ5LmuUN5ordUvBjgbgf/wLazQPbG5Ia8rmhh/Gyptwz9N0ilyQSLx5h7pVc1fl4z+U3O3seFJdO2xI8M; x-main="QKUGsTWzgOIn1MQ?MLW6M7?W9sXe6vT8lOfYIRdLo30O9fDyRRuxqM6Up6QliSso"; at-main=Atza|IwEBIE9-j0Q56LUPug78YuRWyKfqUI-71kqBUKBSv9o0pgXoEdWRTp3QanY2PVmzS5kuRKhJ-2Qf2Y2Xza8aOzNTehTteyzG_d1HL64CDYGBvcFiM3J3K6q-_iEKTHPqIuFGTC7GBQffrkFancRYWCUTNK7ii9qpL1Qvc-lEAmj5xJFsos2I98msXxsTvgVFMUfr1rhJK6CfvkpCAIziz2lPcltlQCWaFsR10_Le_ytyKVYeIZ18m-F1bXhhejn7mTwQJl0; sess-at-main="f/c/JR/xicYEmsJ7sGnJySc4uAQ6ojXuaOFFChXMGho="; _session_id2=64c06a0ae25fee31093af8a3bf3f79e1',
    'If-None-Match': 'W/"087d62f3d9fcbed8836cb66e1e0c2bef"',
    'Referer': 'https://www.goodreads.com/',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Not_A Brand";v="99", "Google Chrome";v="109", "Chromium";v="109"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',}
params = {'ref': 'nav_comm_people',}


directory = os.path.realpath(os.path.join(os.getcwd()))
os.chdir(directory)

with open(os.path.join(directory, 'Data','reviews_urls.json')) as f:
    review_urls = json.load(f)

# THIS MODULE ITERATES OVER ALL USER DATA
# each page contains a list of 20 books that user has read
# We then save each book to the list of books to gather data for

# FIGURE OUT TIMEOUT ERRORS
async def get_user_read_books(session, url, user):
    try:
        async with session.get(url, params=params, cookies=cookies, headers=headers) as response:
            ur_page = await response.text()
            ur_soup = BeautifulSoup(ur_page, 'html.parser')

            # Find what each user has read (as book url)
            user_read = []
            books = ur_soup.find(id='booksBody')

            # ERROR
            if books == None:
                return ([url, user])

            # SAVE
            books = books.find_all('tr')
            for b in books: 
                book_link = b.find('a',href = True)['href']
                user_read.append((book_link,user))
            read_history.extend(user_read)

            # Finalize
            print(f"READ", end=' ')
            return([])
            # return(user_read, None)
    except Exception as e:
        # Handle the exception and log it
        print(f"Error occured while fetching data from {url}: {e}")
        return([url, user])

async def get_read_books(urls_to_view):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60*60*24)) as session:

        tasks = []
        print(tasks)
        for rev_url, user in urls_to_view:
            tasks.append(asyncio.ensure_future(get_user_read_books(session, rev_url, user)))

        failed_urls = await asyncio.gather(*tasks)
        
        failed_urls_return = []
        for url in failed_urls:
            if url != []:
                failed_urls_return.append(url)

        return(failed_urls_return)


#%%
read_history = []
while len(review_urls) > 0:
    review_urls = asyncio.run(get_read_books(review_urls))
    print(review_urls)

with open('read_history.json', 'r') as file:
    json.dump(read_history, file, indent=2)