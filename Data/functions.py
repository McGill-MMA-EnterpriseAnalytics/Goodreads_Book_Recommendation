import aiohttp
import asyncio
import tqdm.asyncio
from bs4 import BeautifulSoup
import re
import numpy as np


async def async_downloader(parser_function: function,  task_list: list) -> tuple(list, list):
    '''This is a generic asyncronous request function for downloading data

    Inputs:
        (1) Downloader Function - this function should take page soup and return 
            relevant data
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
            async_tasks.append(asyncio.ensure_future(page_downloader(session, task, parser_function)))

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



async def page_downloader(session, task_data, parser_function):
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
        url = task_data[0]
        async with session.get(url, params=params, cookies=cookies, headers=headers) as response:
            page = await response.text()
            soup = BeautifulSoup(page, 'html.parser')
            results = parser_function(soup, task_data)
            return (True, results)
    except Exception as e:
        return (False, task_data)


# GET USER IDS AND COUNTRIES
def get_user_list(soup, task_data):
    tab = soup.find("table", {"class": "tableList"})
    results = []
    trs = tab.find_all('tr')
    for t in trs:
        s = t.find_all('td')[1].find('a', href=True)['href']
        user_id = re.findall(r'\d+', s)[0]
        results.append((user_id, task_data[1]))
        
    return(results)
    

def get_user_detail(soup, task_data):
    details = soup.find("div", class_="infoBoxRowItem").text.split(',')
    age = ""
    gender = ""

    for i in details:
        if 'Age ' in i:
            age = re.findall(r'\d+', i)[0]
        if 'Female' in i or 'Male' in i:
            gender = i.strip()
    
    results = [task_data[1], age, gender]
    

def get_page_details(soup, task_data): 
    review_pages = int(soup.find(id='reviewPagination').find_all("a")[-2].text)
    if review_pages > 5:
        check_pages = np.random.choice(review_pages, replace=False, size=5)
    else: 
        check_pages = [i for i in range(review_pages)]

    new_urls = []
    for p in check_pages:
                new_urls.append((f"https://www.goodreads.com/review/list/{task}?page={p+1}",task))
            results.extend(new_urls)