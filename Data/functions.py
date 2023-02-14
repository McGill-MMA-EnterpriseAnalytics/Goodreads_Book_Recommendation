import aiohttp
import asyncio
import tqdm.asyncio
from bs4 import BeautifulSoup
import re
import numpy as np
import json


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
    results = []
    details = soup.find("div", class_="infoBoxRowItem").text.split(',')
    age = ""
    gender = ""

    for i in details:
        if 'Age ' in i:
            age = re.findall(r'\d+', i)[0]
        if 'Female' in i or 'Male' in i:
            gender = i.strip()
    
        results.append(task_data[1], age, gender)
    return(results)
    

def get_page_details(soup, task_data): 
    results = []
    review_pages = int(soup.find(id='reviewPagination').find_all("a")[-2].text)
    if review_pages > 5:
        check_pages = np.random.choice(review_pages, replace=False, size=5)
    else: 
        check_pages = [i for i in range(review_pages)]

    new_urls = []
    for p in check_pages:
        new_urls.append((f"https://www.goodreads.com/review/list/{task_data[1]}?page={p+1}",task_data[1]))
    results.extend(new_urls)
    return(results)
    
    
def get_book_page_details(soup, task_data):
    results = []
    user_read = []
    books = soup.find(id='booksBody')
    books = books.find_all('tr')
    for b in books: 
        book_link = b.find('a',href = True)['href']
        user_read.append((book_link,task_data[1]))
    results.extend(user_read)
    return(results)

    
def book_information(soup, task_data):
    results = []
    book_info = []

    # Find book information
    award = []
    isbn = []

    
    pages_divs = soup.find_all("p", {"data-testid": "pagesFormat"})
    book_pages = pages_divs[0].text
            
    num_of_rating_divs = soup.find_all("span", {"data-testid": "ratingsCount"})
    num_of_rating = re.findall(r'\d+',num_of_rating_divs[0].text)[0]
            
            
    num_of_review_divs = soup.find_all("span", {"data-testid": "reviewsCount"})
    num_of_review = re.findall(r'\d+',num_of_review_divs[0].text)[0]
            

    genre_divs = soup.find_all("a", {"class": "Button Button--tag-inline Button--small"})
    genre = genre_divs[1].text


    publish_divs = soup.find_all("p", {"data-testid":"publicationInfo"})
    publish = publish_divs[0].text

    author_divs = soup.find_all("span",{"class":"ContributorLink__name"})
    author = author_divs[0].text

    title_divs = soup.find_all("h1", {"class": "Text Text__title1"})
    title = title_divs[0].text

    rating_divs = soup.find_all("div", {"class": "RatingStatistics__rating"})
    rating = rating_divs[0].text


    json_str = soup.find_all("script",{'type':'application/ld+json'})[0].string
    data = json.loads(json_str)

    if 'awards' in data.keys():
        award.append(1)
    else :
        award.append(0)

    if 'isbn' in data.keys():
        isbn.append(data.get("isbn"))
    else :
        isbn.append('NA')

    description_divs = soup.find_all("span", {"class": "Formatted"})
    try:
        description = description_divs[0].text
    except IndexError:
        description = "Nil"
            
    book_info.append((task_data[0], book_pages, num_of_rating, num_of_review,
                        genre, publish, author, title, rating, 
                        award, isbn))
    results.extend(book_info)
    
    