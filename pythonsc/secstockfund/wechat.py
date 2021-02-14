
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains

import json
import os
import time


driver = webdriver.Chrome(executable_path="C:/fh/testenv1/chromedriver.exe")
driver.set_page_load_timeout(40)

#Auto chat in wechat by program
try:
    driver.get("https://wx.qq.com") # get接受url可以是如何网址，此处以百度为例
except:
    print(f'{time.ctime()} Loading page timeout.')
    
meimg=driver.find_element(By.CSS_SELECTOR,"img.img")
meimg.click()

mechat=driver.find_element(By.CSS_SELECTOR,"i.web_wechat_tab_launch-chat")
mechat.click()

search=driver.find_element(By.CSS_SELECTOR,"input.frm_search.ng-isolate-scope.ng-pristine.ng-valid")
search.clear()
search.send_keys("Elsa")
search.send_keys(Keys.ENTER)

chat=driver.find_element(By.CSS_SELECTOR,"pre#editArea")
chat.send_keys("Hello world! Testing")
chat.send_keys(Keys.ENTER)

driver.close()

