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

user_list_tasks = itertools.product(*[user_types, user_countries, user_time_frame])
user_list = []
init_failed = 0
while True:
    user_list_data, user_list_fail = asyncio.run(async_downloader(get_user_list, user_list_tasks))
    user_list.extend(user_list_data)

    failed = len(user_list_fail)
    if failed == init_failed:
        break
    init_failed = failed
user_list = list(set(user_list))


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

            results = []
            for t in trs:
                s = t.find_all('td')[1].find('a', href=True)['href']
                user_id = re.findall(r'\d+', s)[0]
                results.append(user_id)

            # Results
            return (True, results)
    except Exception as e:
        return (False, task_data)
