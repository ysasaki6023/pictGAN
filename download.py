# -*- coding: utf-8 -*-
import os,sys,urllib,time
from selenium import webdriver
import argparse

driver = webdriver.Firefox()


def download(url,folder):
    path = os.path.join(folder,os.path.basename(url))
    if os.path.exists(path):
        return
    try:
        img = urllib.urlopen(url)
    except:
        print "failed in urllib:",url
        return
    localfile = open(path, 'wb')
    localfile.write(img.read())
    img.close()
    localfile.close()
    return

def searchAndGet(query,folder):
    driver.get('https://www.google.co.jp/search?q=%s&tbm=isch' % query)

    if not os.path.exists(folder):
        os.makedirs(folder)

    image_urls = set()
    for thumbnail in driver.find_elements_by_css_selector('img.rg_ic'):
        thumbnail.click()
        time.sleep(1)
        for img in driver.find_elements_by_css_selector('img.irc_mi'):
            url = img.get_attribute('src')
            if not url:
                continue
            if not (url.endswith(".png") or url.endswith(".PNG") or url.endswith(".jpg") or url.endswith(".JPG") or url.endswith(".jpeg") or url.endswith(".JPEG")):
                continue
            print (url)
            download(url,folder)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--items","-i",dest="items",type=str,default=None)
    parser.add_argument("--saveFolder","-s",dest="saveFolder",type=str,default="img")
    args = parser.parse_args()

    if not args.items:
        print "usage:  python download.py -i dog,cat"
        sys.exit(-1)

    for item in args.items.split(","):
        searchAndGet(query="icons %s"%item  ,folder=os.path.join(args.saveFolder,item))
#searchAndGet(query="icons"  ,folder="img/general")
#searchAndGet(query="icons robot"  ,folder="img/robot")
#searchAndGet(query="icons dog"    ,folder="img/dog")
#searchAndGet(query="icons cat"    ,folder="img/cat")
#searchAndGet(query="icons man"    ,folder="img/man")
#searchAndGet(query="icons woman"  ,folder="img/woman")
#searchAndGet(query="icons rocket" ,folder="img/rocket")
