# %% IMPORTS
from __future__ import annotations
import os
import re
import json
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import datetime
import requests
import tqdm
import tqdm.asyncio
import aiohttp
import asyncio
import itertools
import nest_asyncio
nest_asyncio.apply()


# CONFIG
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
           '_session_id2': '64c06a0ae25fee31093af8a3bf3f79e1', }
headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,/;q=0.8,application/signed-exchange;v=b3;q=0.9',
           'Accept-Language': 'en-US,en;q=0.9',
           'Cache-Control': 'max-age=0',
           'Connection': 'keep-alive',
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
           'sec-ch-ua-platform': '"macOS"', }
params = {'ref': 'nav_comm_people', }


# DOWNLOADER FUNC
async def async_downloader(downloader: function, task_list: list) -> tuple(list, list):
    '''This is a generic asyncronous request function for downloading data

    Inputs:
        (1) Downloader Function - this function should return whether it was
            successful or not and the corresponding data
        (2) Task List - each item in the list should be enough to execute
            the downloader function

    Outputs:
        (1) Results - results list
        (2) Failed Tasks - a subset from initial task list indicating
            which tasks failed'''
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60*60*24)) as session:

        # Initializes tasks
        async_tasks = []
        for task in task_list:
            async_tasks.append(asyncio.ensure_future(downloader(session, task)))

        # Executes Tasks
        task_res = await tqdm.asyncio.tqdm_asyncio.gather(*async_tasks)

        # Successful Tasks
        successful_task_data = []
        for tid in range(len(task_res)):
            if task_res[tid][0] == True:
                successful_task_data.extend(task_res[tid][1])

        # Failed Tasks
        failed_task_data = []
        for tid in range(len(task_res)):
            if task_res[tid][0] == False:
                successful_task_data.append(task_res[tid][1])

        return (successful_task_data, failed_task_data)


# GET USER LIST
async def get_user_list(session, task_data):
    '''Downloads a list of user

    Inputs:
        (1) Session - async requirements
        (2) Task Data - tuple of (User Types, Countries, Time Frames)

    Outputs:
        (1) Bool - whether a task is successful or not
        (2) List or Value
                - if sucessful returns list of data points
                - if not sucessful returns failed task data '''
    try:
        url = task_data[0] + task_data[1] + task_data[2]
        async with session.get(url, params=params, cookies=cookies, headers=headers) as response:
            # Request
            page = await response.text()
            soup = BeautifulSoup(page, 'html.parser')

            # Parse
            tab = soup.find("table", {"class": "tableList"})
            if tab is None:
                return (True, [])
            results = []
            trs = tab.find_all('tr')
            for t in trs:
                s = t.find_all('td')[1].find('a', href=True)['href']
                user_id = re.findall(r'\d+', s)[0]
                results.append(user_id)

            # Results
            return (True, results)
    except Exception as e:
        return (False, task_data)

user_types = ['https://www.goodreads.com/user/best_reviewers',
              'https://www.goodreads.com/user/top_reviewers',
              'https://www.goodreads.com/user/top_readers']
user_countries = ['?country=CA', '?country=all', '?country=US', '?country=US', '?country=AU',
                  '?country=SC', '?country=NZ', '?country=IE', '?country=GB',
                  '?country=SG', '?country=GB', '?country=DO', '?country=TT',
                  '?country=MT', '?country=BB', '?country=LC', '?country=GY']
user_time_frame = ['&duration=w', '&duration=m', '&duration=y', '&duration=a']
user_list_tasks = []
for comb in itertools.product(*[user_types, user_countries, user_time_frame]):
    user_list_tasks.append((comb[0]+comb[1]+comb[2], comb[1].split('=')[1]))
user_list_tasks
# user_list = []
# failed = 0
# while True:
#     user_list_data, user_list_fail = asyncio.run(async_downloader(get_user_list, user_list_tasks))
#     user_list.extend(user_list_data)

#     if len(user_list_fail) == failed:
#         break
#     failed = len(user_list_fail)
# user_list = list(set(user_list))

#%%
# GET USER DETAILS
async def get_user_detail(session, task_data):
    '''Downloads a list of user

    Inputs:
        (1) Session - async requirements
        (2) Task Data - tuple of (User Types, Countries, Time Frames)

    Outputs:
        (1) Bool - whether a task is successful or not
        (2) List or Value
                - if sucessful returns list of data points
                - if not sucessful returns failed task data '''
    try:
        url = "https://www.goodreads.com//user/show/"+task_data
        async with session.get(url, params=params, cookies=cookies, headers=headers) as response:
            # Request
            page = await response.text()
            soup = BeautifulSoup(page, 'html.parser')

            # Parse
            details = soup.find("div", class_="infoBoxRowItem").text
            if details is None:
                return (False, task_data)

            details = details.split(',')
            age = ""
            gender = ""

            for i in details:
                if 'Age ' in i:
                    age = re.findall(r'\d+', i)[0]
                if 'Female' in i or 'Male' in i:
                    gender = i.strip()
            
            user_deets = [user_id, age, gender]

        

       

# GET USER PAGE URLS
    '''Downloads two pages of lists of books that each user has read

    Inputs:
        (1) Session - async requirements
        (2) Task Data - tuple of (User IDs)

    Outputs:
        (1) Bool - whether a task is successful or not
        (2) List or Value
                - if sucessful returns list of data points
                - if not sucessful returns failed task data '''

async def get_user_review_pages(session, task_data):
    async with session.get("https://www.goodreads.com/review/list/"+task_data[0],
                                 params=params, cookies=cookies, headers=headers) as response:
        try: 
            ur_page = await response.text()
            ur_soup = BeautifulSoup(ur_page, 'html.parser')
            review_pages = int(ur_soup.find(id='reviewPagination').find_all("a")[-2].text)
            if review_pages > 5:
                check_pages = np.random.choice(review_pages, replace=False, size=5)
            else: 
                check_pages = [i for i in range(review_pages)]

            new_urls = []
            for p in check_pages:
                new_urls.append((f"https://www.goodreads.com/review/list/{task_data[0]}?page={p+1}",task_data[0]))
            results.extend(new_urls)
            return (True, results)
        
     
        except Exception as e:
            return (False, task)
    
async def download_comments(task_data):
    async with aiohttp.ClientSession(timeout = aiohttp.ClientTimeout(total=60*60*24)) as session:
        tasks = []
        for task in task_data:
            tasks.append(asyncio.ensure_future(get_user_review_pages(session, task_data)))

        failed_users = await asyncio.gather(*tasks)
        failed_users_res = []
        for f_user in failed_users:
            if f_user != "":
                failed_users_res.append(f_user)

        return(failed_users_res)

results = []
failed_users = asyncio.run(download_comments())


# GET BOOK URLS
'''Downloads pages of all books in the two lists of books that each user has read

    Inputs:
        (1) Session - async requirements
        (2) url - tuple of book url on goodreads
        (3) user - User ID of user that read the book

    Outputs:
        (1) Bool - whether a task is successful or not
        (2) List or Value
                - if sucessful returns list of data points
                - if not sucessful returns failed task data '''


async def get_user_read_books(session, url, user):
    try:
        async with session.get(url, params=params, cookies=cookies, headers=headers) as response:
            ur_page = await response.text()
            ur_soup = BeautifulSoup(ur_page, 'html.parser')

            # Find what each user has read (as book url)
            user_read = []
            books = ur_soup.find(id='booksBody')

            # SAVE
            books = books.find_all('tr')
            for b in books: 
                book_link = b.find('a',href = True)['href']
                user_read.append((book_link,user))
            read_history.extend(user_read)

            return(True, read_history)

    except Exception as e:
        # Handle the exception and log it
        # print(f"Error occured while fetching data from {url}: {e}")
        return(False, url, user)

async def get_read_books(urls_to_view):
    async with aiohttp.ClientSession(trust_env=True,timeout=aiohttp.ClientTimeout(total=60*60*24)) as session:

        tasks = []
        for rev_url, user in urls_to_view:
            tasks.append(asyncio.ensure_future(get_user_read_books(session, rev_url, user)))

        # failed_urls = await asyncio.gather(*tasks)
        failed_urls = await tqdm.asyncio.tqdm_asyncio.gather(*tasks)
        
        failed_urls_return = []
        for url in failed_urls:
            if url != []:
                failed_urls_return.append(url)

        return(failed_urls_return)

tot_count = 0
read_history = []
while len(review_urls) > 0:
    prev_len = len(review_urls) 
    review_urls = asyncio.run(get_read_books(review_urls))
    print(len(review_urls))

    with open('users_bookshelves_kv.json', 'w') as file:
        json.dump(list(read_history), file)
    if len(review_urls) == prev_len:
        break



async def get_books(session, url):
    url = 'https://www.goodreads.com' + url
    try:
        async with session.get(url, params=params, cookies=cookies, headers=headers) as response:
            ur_page = await response.text()
            ur_soup = BeautifulSoup(ur_page, 'html.parser')
            book_info = []

            # Find book information
            award = []
            isbn = []

    
            pages_divs = ur_soup.find_all("p", {"data-testid": "pagesFormat"})
            if pages_divs == None:
                return url
            book_pages = pages_divs[0].text
            
            num_of_rating_divs = ur_soup.find_all("span", {"data-testid": "ratingsCount"})
            if num_of_rating_divs == None:
                return url
            num_of_rating = re.findall(r'\d+',num_of_rating_divs[0].text)[0]
            
            
            num_of_review_divs = ur_soup.find_all("span", {"data-testid": "reviewsCount"})
            if num_of_review_divs == None:
                return url
            num_of_review = re.findall(r'\d+',num_of_review_divs[0].text)[0]
            


            genre_divs = ur_soup.find_all("a", {"class": "Button Button--tag-inline Button--small"})
            if genre_divs == None:
                return url
            genre = genre_divs[1].text


            publish_divs = ur_soup.find_all("p", {"data-testid":"publicationInfo"})
            if publish_divs == None:
                return url
            publish = publish_divs[0].text

            author_divs = ur_soup.find_all("span",{"class":"ContributorLink__name"})
            if author_divs == None:
                return url
            author = author_divs[0].text



            title_divs = ur_soup.find_all("h1", {"class": "Text Text__title1"})
            if title_divs == None:
                return url
            title = title_divs[0].text


            rating_divs = ur_soup.find_all("div", {"class": "RatingStatistics__rating"})
            if rating_divs == None:
                return url
            rating = rating_divs[0].text


            json_str = ur_soup.find_all("script",{'type':'application/ld+json'})[0].string
            if json_str == None:
                return url
            data = json.loads(json_str)


            if 'awards' in data.keys():
                award.append(1)
            else :
                award.append(0)

            if 'isbn' in data.keys():
                isbn.append(data.get("isbn"))
            else :
                isbn.append('NA')


            description_divs = ur_soup.find_all("span", {"class": "Formatted"})
            try:
                description = description_divs[0].text
            except IndexError:
                description = "Nil"
            
            book_info.append((url, book_pages, num_of_rating, num_of_review,
                              genre, publish, author, title, rating, 
                              award, isbn))
            all_books.extend(book_info)

            # Finalize
            # print(f"READ", end=' ')
            return([])

    except Exception as e:
        # Handle the exception and log it
        # print(f"Error occured while fetching data from {url}: {e}")
        return([url])
    

async def get_books_async(urls_to_view):
    async with aiohttp.ClientSession(trust_env=True,timeout=aiohttp.ClientTimeout(total=60*60*24)) as session:

        tasks = []
        for rev_url in urls_to_view:
            tasks.append(asyncio.ensure_future(get_books(session, rev_url)))

        failed_urls = await tqdm.asyncio.tqdm_asyncio.gather(*tasks)
        
        failed_urls_return = []
        for url in failed_urls:
            if url != []:   
                failed_urls_return.extend(url)

        return(failed_urls_return)


all_books = []
while len(books) > 0:
    init_len = len(books)
    books = asyncio.run(get_books_async(books))

    if init_len == len(books):
        break
        
    


# GET BOOK INFORMATION 
async def get_books(session, url):
    url = 'https://www.goodreads.com' + url
    try:
        async with session.get(url, params=params, cookies=cookies, headers=headers) as response:
            ur_page = await response.text()
            ur_soup = BeautifulSoup(ur_page, 'html.parser')
            book_info = []

          

            # Find book information
            award = []
            isbn = []

            pages_divs = ur_soup.find_all("p", {"data-testid": "pagesFormat"})
            book_pages = pages_divs[0].text
                        
            num_of_rating_divs = ur_soup.find_all("span", {"data-testid": "ratingsCount"})
            num_of_rating = (','.join(re.findall(r'\d+',num_of_rating_divs[0].text)))
                        
                        
            num_of_review_divs = ur_soup.find_all("span", {"data-testid": "reviewsCount"})
            num_of_review = (','.join(re.findall(r'\d+',num_of_review_divs[0].text)[0]))
            

            genres = set()
            genre_divs = ur_soup.find_all("a", {"class": "Button Button--tag-inline Button--small"})
            for genre in genre_divs[0:6]: 
                genres.add(genre.text.lower())



            publish_divs = ur_soup.find_all("p", {"data-testid":"publicationInfo"})
            publish = publish_divs[0].text

            authors = set()
            author_divs = ur_soup.find_all("div",{"class":"ContributorLinksList"})
            for author in author_divs[0].find_all('a', class_='ContributorLink'):
                #print(author.find_all("span",{"data-testid":"role"},{"class":"ContributorLink__role"}))
                print(author.find("span",{"class":"ContributorLink__name"}))
                print(author.find("span",{"data-testid":"role"},{"class":"ContributorLink__role"}))

                if author.find("span",{"data-testid":"role"},{"class":"ContributorLink__role"}) == None:
                    authors.add(author.find("span",{"class":"ContributorLink__name"}).text)
  
  
 

            title_divs = ur_soup.find_all("h1", {"class": "Text Text__title1"})
            title = title_divs[0].text


            rating_divs = ur_soup.find_all("div", {"class": "RatingStatistics__rating"})
            rating = rating_divs[0].text


            json_str = ur_soup.find_all("script",{'type':'application/ld+json'})[0].string
            data = json.loads(json_str)


            if 'awards' in data.keys():
                award.append(1)
            else :
                award.append(0)

            if 'isbn' in data.keys():
                isbn.append(data.get("isbn"))
            else :
                isbn.append('NA')


            description_divs = ur_soup.find_all("span", {"class": "Formatted"})
            try:
                description = description_divs[0].text
            except IndexError:
                description = "Nil"
        
            book_info.append((url, book_pages, num_of_rating, num_of_review,
                            genres, publish, authors, title, rating, 
                            award, isbn))
            all_books.extend(book_info)

            # Finalize
            # print(f"READ", end=' ')
            return([])

    except Exception as e:
        # Handle the exception and log it
        # print(f"Error occured while fetching data from {url}: {e}")
        return([url])
    

async def get_books_async(urls_to_view):
    async with aiohttp.ClientSession(trust_env=True,timeout=aiohttp.ClientTimeout(total=60*60*24)) as session:

        tasks = []
        for rev_url in urls_to_view:
            tasks.append(asyncio.ensure_future(get_books(session, rev_url)))

        failed_urls = await tqdm.asyncio.tqdm_asyncio.gather(*tasks)
        
        failed_urls_return = []
        for url in failed_urls:
            if url != []:   
                failed_urls_return.extend(url)

        return(failed_urls_return)


all_books = []
while len(books) > 0:
    init_len = len(books)
    books = asyncio.run(get_books_async(books))

    if init_len == len(books):
        break