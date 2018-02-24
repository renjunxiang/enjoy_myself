from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import pandas as pd
import numpy as np
import time

# 设置项
profile = webdriver.FirefoxProfile()
profile.set_preference('permissions.default.image', 2)  # 无图模式
profile.set_preference('browser.migration.version', 9001)  # 部分需要加上这个

options = Options()
options.add_argument('-headless') #后台模式

driver = webdriver.Firefox(firefox_profile=profile,firefox_options=options)

# 查找城市
driver.get("https://www.lagou.com/gongsi/allCity.html?option=0-0-0")
city_raw = driver.find_elements_by_xpath("//table[@class='word_list']/tbody")[0].text
city_raw = city_raw.split('\n')
city_raw = [i.split(' ') for i in city_raw if len(i) > 1]
cities = []
for i in city_raw:
    cities += i


def region(region_name='上海'):
    print('=====开始爬取 %s=====' % region_name)
    # 跳转至该地区公司
    driver.get("https://www.lagou.com/gongsi/allCity.html?option=0-0-0")
    driver.find_elements_by_xpath("//li/a[text()='%s']" % region_name)[0].click()
    page = 0
    info = []
    while True:
        # 爬取公司信息
        companys_info = driver.find_elements_by_xpath("//li[@class='company-item']")

        for company_info in companys_info:
            text = company_info.text.split('\n')
            text += [region_name]
            info.append(np.array(text)[[0, 1, 2, 3, 4, 7, 9]])
        try:
            time.sleep(2)
            driver.find_elements_by_xpath("//span[@class='pager_next ']")[0].click()
            page += 1
            print('爬取 第%d页' % page)
            print(len(info))
        except:
            break
    return info


infoes = []
n=0
for city in cities:
    infoes += region(region_name=city)
    n+=1
    print('完成爬取  %s,进度 %d/%d' % (city,n,len(cities)))
