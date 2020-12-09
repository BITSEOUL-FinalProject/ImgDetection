
import urllib
from bs4 import BeautifulSoup as bs
from urllib.parse import urlencode, quote_plus, unquote
from urllib.request import urlopen, urlretrieve
import urllib
import os
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

base_url = 'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query='  
plusUrl = input('검색어 입력: ') 
url = base_url + quote_plus(plusUrl) + '%EC%B6%9C%EC%97%B0%EC%A7%84' 

html = urlopen(url)
soup = bs(html, "html.parser")
name = soup.find("div", class_="list_image_info _content").find_all("li")

find_imglist= list()
find_casting = list()
find_namelist = list()

for item in name:
    find_name = item.find_all(class_="_text")[1] #주인공 이름
    find_namelist.append(find_name.get_text())

    find_img = item.find(class_='item').find_all(class_='thumb')
    for j in find_img:
            img = j.find('img')
            find_imglist.append(img.get('src'))
            find_casting.append(img.get('alt'))

# find_imglist = np.array(find_imglist)
# find_casting = np.array(find_casting)
# find_namelist = np.array(find_namelist)

# np.save('./MJK/data/npy/find_imglist.npy', arr=find_imglist)
# np.save('./MJK/data/npy/find_casting.npy', arr=find_casting)
# np.save('./MJK/data/npy/find_namelist.npy', arr=find_namelist)


for i in range(2):
    path = './teamproject/images2/'+str(i)+'/'
    os.makedirs(path, exist_ok=True)
    driver = webdriver.Chrome(r"D:\workspace\Study\teamProject\chromedriver.exe")
    driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl")
    elem = driver.find_element_by_name("q")
    elem.send_keys(find_namelist[i])
    elem.send_keys(Keys.RETURN)

    SCROLL_PAUSE_TIME = 1
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                driver.find_element_by_css_selector(".mye4qd").click()
            except:
                break
        last_height = new_height

    images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
    count = 1
    for image in images:
        try:
            image.click()
            time.sleep(1)
            imgUrl = driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img').get_attribute("src")
            opener=urllib.request.build_opener()
            opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(imgUrl, path + str(count) + ".jpg")
            count = count + 1
        except:
            pass