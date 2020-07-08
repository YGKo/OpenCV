import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from apscheduler.schedulers.blocking import BlockingScheduler

#http://www.cgv.co.kr//common/showtimes/iframeTheater.aspx?areacode=01&amp;theatercode=0013&amp;date=20200708

def job_function():
    # HTTP GET Requesth
    req = requests.get('http://www.cgv.co.kr//common/showtimes/iframeTheater.aspx?areacode=01&theatercode=0001&date=20200710&screencodes=&screenratingcode=&regioncode=')
    ## HTML 소스 가져오기
    html = req.text
    ## 이 글에서는 Python 내장 html.parser를 이용했다.
    soup = BeautifulSoup(req.content, 'html.parser')
    title_list = soup.select('div.info-movie')
    for i in title_list:
        print(i.select_one('a > strong').text.strip())
    is_4D = soup.select_one('span.forDX')
    is_imax = soup.select_one('span.imax')
    if (is_imax):
        is_imax = is_imax.find_parent('div', class_='col-times')
        title = is_imax.select_one('div.info-movie > a > strong').text.strip()
        print(title+'IMAX가 개봉했습니다')
        sche.pause()
    if (is_4D):
        is_4D = is_4D.find_parent('div', class_='col-times')
        title = is_4D.select_one('div.info-movie > a > strong').text.strip()
        print(title+'4DX가 개봉했습니다')
        sche.pause()

sche = BlockingScheduler()
sche.add_job(job_function, 'interval', seconds=5)
sche.start()
