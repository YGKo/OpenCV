import requests
from bs4 import BeautifulSoup
from selenium import webdriver
#pprint는 딕셔너리의 데이터가 긴 경우에 좀 더 보기 편하게 보여주게 도와준다.
from pprint import pprint

# 내려받은 chromedriver의 위치
driver = webdriver.Chrome('./driver/chromedriver')
# 웹 자원 로드를 위해 3초까지 기다린다.
driver.implicitly_wait(3)
# url접근
driver.get('https://www.weather.go.kr/weather/forecast/timeseries.jsp')

#driver.find_element_by_name('selectedLocalName').send_keys('경기도 고양시덕양구 행신1동')

driver.find_element_by_xpath("//*[@id='content_weather']/div[2]/dl/dd/a[1]").click()
driver.implicitly_wait(1)
driver.execute_script('document.getElementById("selectedLocalName").removeAttribute("readonly")')
driver.find_element_by_xpath("//*[@id='selectedLocalName']").clear()
driver.find_element_by_id("selectedLocalName").send_keys("경기도 고양시덕양구 행신1동")
driver.implicitly_wait(3)
driver.find_element_by_xpath("//*[@id='layor_area']/form/fieldset/p/a[2]/img").click()
#driver.execute_script ("arguments[0].click();",btn)

# HTTP GET Requesth
req = requests.get('https://www.weather.go.kr/weather/forecast/timeseries.jsp')
## HTML 소스 가져오기
html = req.text
#print("=======HTML Page==============",html)

## 이 글에서는 Python 내장 html.parser를 이용했다.
soup = BeautifulSoup(req.content, 'html.parser')
#print(soup.p)
#print(soup.p.string)
#print(soup.p.string)
#print(soup.h1)
#print(soup.find_all("table"))
#print(soup.find_all(attrs={'class':'forecastNew3'}))
#print(soup.select('table'))
#for child in soup.ul.children:
#    print(child)
#for parent in soup.ul.parents:
#    print(parent)
table = soup.find("table", class_="forecastNew3")
tr = table.tbody.tr
for t in tr.children:
	if t.name == 'th':
		if t['scope'] == 'colgroup':
			num = int(t['colspan'])
			for i in range(num):
				print(t.get_text(), end = ' ')

tr = tr.next_sibling.next_sibling
print('@시각')
for t in tr.children:
	if t.name == 'td':
		for i in t.contents:
			if i.name =='p':
				print(i.get_text(), end=' ')
print('\n')

tr = tr.next_sibling.next_sibling
print('@날씨')
for w in tr.children:
    if w.name == 'td' and len(w.contents) > 0:
        print(w['title'], end=' ')
print('\n')

tr = tr.next_sibling.next_sibling
print('@강수 확률(%)')
for w in tr.children:
    if w.name == 'td' and len(w.contents) > 0:
        print(w.contents[0], end=' ')
print('\n')

tr = tr.next_sibling.next_sibling
print('@강수량(mm)')
for w in tr.children:
    if w.name == 'td' and len(w.contents) > 0:
        num = int(w['colspan'])
        for i in range(num):
            print(w.contents[0].strip(), end=' ')
print('\n')

tr = tr.next_sibling.next_sibling
print('@최저/최고 기온(℃)')
for w in tr.children:
	if w.name == 'td' and len(w.contents) > 0:
		num = int(w['colspan'])
		for i in range(num):
			print(w.contents[0].get_text(), end='/')
			print(w.contents[2].get_text(), end=' ')
print('\n')

tr = tr.next_sibling.next_sibling
print('@기온(℃)')
for w in tr.children:
	if w.name == 'td' and len(w.contents) > 0:
		print(w.contents[0].get_text(), end=' ')
print('\n')

tr = tr.next_sibling.next_sibling
print('@풍향/풍속(km/h)')
for w in tr.children:
	if w.name == 'td' and len(w.contents) > 0:
		print(w['title'], end= ' ')
print('\n')

tr = tr.next_sibling.next_sibling
print('@습도(%)')
for w in tr.children:
	if w.name == 'td' and len(w.contents) > 0:
		print(w.contents[0].get_text(), end=' ')
print('\n')
