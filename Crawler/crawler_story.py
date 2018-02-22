from TextClustering.TextClustering import TextClustering
import pandas as pd
import os
import numpy as np
from selenium import webdriver
import time
import threading
from multiprocessing import Process, Lock, Queue
from openpyxl import load_workbook

# DIR = os.path.dirname(__file__)

DIR = 'D:\\github\\enjoy_myself\\crawler'

# 设置项
profile = webdriver.FirefoxProfile()
profile.set_preference('permissions.default.image', 2)  # 无图模式
profile.set_preference('browser.migration.version', 9001)  # 部分需要加上这个

def crawler(class_name, lock):
    '''
    因为网站有编码问题，直接保存csv或者excel会报错，因此存为npy格式
    :param class_name: 
    :param lock: 
    :return: 
    '''
    # 进入主页
    driver = webdriver.Firefox(firefox_profile=profile)
    driver.get("http://www.xqishu.com/")

    driver.find_elements_by_xpath("//div[@class='nav']/a[text()='%s']" % class_name)[0].click()  # 点击分类
    time.sleep(1)
    print('=====begin crawler %s=====' % class_name)
    window_handle_now = driver.window_handles[1]  # 查询新开页句柄
    driver.close()
    driver.switch_to.window(window_handle_now)  # 跳转至新开页

    titles = []
    urls = []
    infoes = []
    stars = []
    describtions = []
    page = 0

    f = open(DIR + '/log.txt', mode='a')
    judge = 1
    while judge > 0:
        # 对每个内容进行最多三次判断，爬取失败则跳过
        counts = 0
        while True:
            onepage_title = driver.find_elements_by_xpath("//div[@class='listBox']/ul/li/a")  # 名称
            title = [j.text for j in onepage_title]
            counts += 1
            if counts > 3:
                break
            if title != []:
                break
        titles += title

        counts = 0
        while True:
            onepage_url = driver.find_elements_by_xpath("//div[@class='listBox']/ul/li/a")  # 网址
            url = [j.get_attribute('href') for j in onepage_url]
            counts += 1
            if counts > 3:
                break
            if url != []:
                break
        url += url

        counts = 0
        while True:
            onepage_info = driver.find_elements_by_xpath("//div[@class='s']")  # 基本信息
            info = [[k.split('：')[1] for k in np.array(j.text.split('\n'))[[0, 1, 3]]] for j in onepage_info]
            counts += 1
            if counts > 3:
                break
            if info != []:
                break
        infoes += info

        counts = 0
        while True:
            onepage_star = driver.find_elements_by_xpath("//div[@class='s']/em")  # 评分
            star = [float(j.get_attribute('class').split('lstar')[1]) for j in onepage_star]
            counts += 1
            if counts > 3:
                break
            if star != []:
                break
        stars += star

        counts = 0
        while True:
            onepage_describtion = driver.find_elements_by_xpath("//div[@class='u']")  # 内容简介
            describtion = [(j.text) for j in onepage_describtion]
            counts += 1
            if counts > 3:
                break
            if describtion != []:
                break
        describtions += describtion

        page += 1
        judge += 1
        if page % 20 == 0:
            print('-----finish page %d-----' % page)
            titles1 = pd.DataFrame(titles, columns=['title'])
            urls1 = pd.DataFrame(urls, columns=['url'])
            stars1 = pd.DataFrame(stars, columns=['stars'])
            infoes1 = pd.DataFrame(infoes, columns=['author', 'size', 'update'])
            describtions1 = pd.DataFrame(describtions, columns=['describtions'])
            #每20页保存一次
            info_all1 = pd.concat([titles1, urls1, infoes1, describtions1, stars1], axis=1)
            info_all1.to_csv(DIR + '/%s.csv' % (class_name + str(page)), index=False, encoding='gbk')
            np.save(DIR + '/%s.npy' % (class_name + str(page)), info_all1)

        try:
            driver.find_elements_by_xpath("//div[@class='tspage']/a[text()='下一页']")[0].click()  # 下一页
            time.sleep(1)
        except:
            judge = -1
    # 关闭尾页
    print('=====finish crawler: %s=====' % class_name)
    driver.close()

    # 拼接数据
    titles = pd.DataFrame(titles, columns=['title'])
    urls = pd.DataFrame(urls, columns=['url'])
    stars = pd.DataFrame(stars, columns=['stars'])
    infoes = pd.DataFrame(infoes, columns=['author', 'size', 'update'])
    describtions = pd.DataFrame(describtions, columns=['describtions'])

    lock.acquire()
    print(lock)
    try:
        info_all = pd.concat([titles, urls, infoes, describtions, stars], axis=1)
        info_all['classify'] = class_name
        np.save(DIR + '/%s.npy' % (class_name), info_all)
        print('=====%s has been writen into excel=====' % class_name)
        f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) +
                '  %s has been writen into excel' % class_name + '\n')
        print('finish put %s' % class_name)
    finally:
        lock.release()
        print('%s lock release' % class_name)
        f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) +
                '  %s lock release' % class_name + '\n')
        f.close()


if __name__ == "__main__":
    # 进入主页
    driver = webdriver.Firefox(firefox_profile=profile)
    driver.get("http://www.xqishu.com/")

    # 读取分类信息
    class_all = driver.find_elements_by_xpath("//div[@class='nav']/a")
    class_1 = [i.text for i in class_all][2:]
    driver.close()
    lock = Lock()  # 加入进程锁
    p_lst = []
    for name_1 in class_1:
        po = Process(target=crawler, kwargs={'class_name': name_1,
                                             'lock': lock
                                             })
        p_lst.append(po)  # 创建子进程
    # 启动子进程
    for p in p_lst:
        p.start()
    # 等待子进程全部结束
    for p in p_lst:
        p.join()
    print('end')
