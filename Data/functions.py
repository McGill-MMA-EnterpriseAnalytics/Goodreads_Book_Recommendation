import aiohttp
import asyncio
import tqdm.asyncio
import BeautifulSoup


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
            results = []
            trs = tab.find_all('tr')
            for t in trs:
                s = t.find_all('td')[1].find('a', href=True)['href']
                user_id = re.findall(r'\d+', s)[0]
                results.append((user_id, task_data[1]))

            # Results
            return (True, results)
    except Exception as e:
        return (False, task_data)
