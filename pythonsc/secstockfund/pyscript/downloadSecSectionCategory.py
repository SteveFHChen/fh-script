#Download stock/fund category info
from fhsechead import *
import execjs

currentDate = time.strftime("%Y%m%d", time.localtime())
savePath="C:/fh/testenv1/sec/"

logFile='C:/fh/testenv1/sec/stock.log'
fLog = LogWriter(logFile)

driver = webdriver.Chrome(executable_path=chromeDriverExe, chrome_options=chrome_options)
driver.set_page_load_timeout(40)

batchSize = 100

def captureFenghuangSecSectionCategory(dataFile, forceRecover=False):
    if forceRecover==False:
        if os.path.exists(dataFile)==True:
            fLog.writeLog(f"Data file already exists. {dataFile}")
            return
    else:
        fLog.writeLog(f"Data file not exists, capturing from internet...")
    
    try:
        driver.get("http://app.finance.ifeng.com/hq/list.php?type=stock_a&class=zs") 
    except:
        fLog.writeLog(f'Loading page timeout.')

    jsSecTypeList = """
        secTypeList = [];
        document.querySelectorAll("div.label li a").forEach(function(t, i, a){
            secTypeList.push({
                "secType": t.innerText,
                "secTypeHref": t.href
            });
        });
        return JSON.stringify(secTypeList);
        """
    secTypeList=json.loads(driver.execute_script(jsSecTypeList))

    for secType in secTypeList:
        #secType = secTypeList[0]
        
        if secType['secType']=='基金':
            continue
        
        driver.get(secType['secTypeHref']) 
        jsSecSubTypeList = f"""
                secType = "{secType['secType']}";
                secSubTypeList = [];
                document.querySelectorAll("div.result p a").forEach(function(st, i, a){{
                    secSubTypeList.push({{
                        "secType": secType,
                        "secSubType": st.innerText,
                        "secSubTypeHref": st.href
                    }});
                }});
                
                return JSON.stringify(secSubTypeList);
                """
        secSubTypeList=json.loads(driver.execute_script(jsSecSubTypeList))
        secType['secSubType'] = secSubTypeList
        
        for secSubType in secSubTypeList:
            #secSubType =secSubTypeList[0]
            driver.get(secSubType['secSubTypeHref'])
            jsSecObjectList = """
                    secObjectList = [];
                    document.querySelectorAll("div.result ul li a").forEach(function(s, i, a){
                        title = s.innerText;
                        startIndex = title.lastIndexOf("(")
                        endIndex = title.lastIndexOf(")")
                        code = title.slice(startIndex+1, endIndex)
                        name = title.slice(0, startIndex)
                        secObjectList.push({
                            "code": code,
                            "name": name
                        });
                    });
                    
                    return JSON.stringify(secObjectList);
                    """
            secObjectList=json.loads(driver.execute_script(jsSecObjectList))
            secSubType['object'] = secObjectList
            
    
    f=open(dataFile, 'w')

    #Way 1 - write:
    #f.writelines([str(secTypeList)])
    # key and value will be enclosed by double quotes

    #Way 2 - write:
    #json.dump(secTypeList, f) #Chinese will be saved as unicode.
    json.dump(secTypeList, f, ensure_ascii=False) #File format will be GB2312
    # key and value will be enclosed by single quotes

    f.flush()
    f.close()

def updateFenghuangSecSectionCategory(dataFile):
    f1 = open(dataFile, 'r')

    #Way 1 read - parse to dictionary failed
    #s1 = f1.readlines()
    #json.loads(s1)
    #Expecting property name enclosed in double quotes

    #Way 2 write:
    s1 = json.load(f1)

    f1.close()

    for secType in s1:
        fLog.writeLog(secType['secType'])
        
        if secType['secType']=="基金":
            continue
        
        for secSubType in secType['secSubType']:
            fLog.writeLog(f"|----{secSubType['secSubType']}[{len(secSubType['object'])}]")
            
            successIndicate = 0
            '''
            #Low performance
            for obj in secSubType['object']:
                #obj = secSubType['object'][0]
                sqlIndicate = f"""
                    UPDATE sec_stock 
                    SET section_name='{secType["secType"]}', subsection_name='{secSubType["secSubType"]}' 
                    WHERE stock_code='{obj["code"]}'
                    """
            '''
            #Use batch update DB data to improve performance.
            sqlIndicate = Nonel
            for batchi in range(math.ceil(len(secSubType['object'])/batchSize)):
                objects = secSubType['object'][batchi*batchSize: (batchi+1)*batchSize]
                objectsStr = "'"+"','".join([obj['code'] for obj in objects])+"'"
                if secType['secType'] =='A股' and secSubType["secSubType"]=='指数':
                    sqlIndicate = f"""
                        UPDATE sec_stock 
                        SET section_name='{secType["secType"]}', subsection_name='{secSubType["secSubType"]}' 
                        WHERE region_code='CN' AND (stock_type='指数' OR section_name IS NULL) AND stock_code IN ({objectsStr})
                        """
                else:
                    sqlIndicate = f"""
                        UPDATE sec_stock 
                        SET section_name='{secType["secType"]}', subsection_name='{secSubType["secSubType"]}' 
                        WHERE (stock_type!='指数' OR section_name IS NULL) and stock_code in ({objectsStr})
                        """
                print(sqlIndicate)
                rowcount = cs1.execute(sqlIndicate)
                successIndicate = successIndicate + rowcount
            conn.commit()
            fLog.writeLog(f"     Success indicated [{successIndicate}] objects, no update [{len(secSubType['object']) - successIndicate}].")
            #time.sleep(5)

#Capture stock section and category, and fill into DB
def captureSohuSecSectionCategory(dataFile, forceRecover=False):
    if forceRecover==False:
        if os.path.exists(dataFile)==True:
            print("Data file already exists. {dataFile}")
            return
    else:
        fLog.writeLog(f"Data file not exists, capturing from internet...")
    
    try:
        driver.get("https://q.stock.sohu.com/cn/bk.shtml")
    except:
        fLog.writeLog(f'Loading page timeout, company is {company}.')
        
    stockSectionCategoryList={"industry": NULL, "region": NULL, "concept": NULL} #"shared": NULL, 
    sectionListLi=driver.find_elements(By.CSS_SELECTOR,"ul#BIZ_MS_phTab li")
    for sectionLi in sectionListLi:
        sectionLi.click()
        time.sleep(3)

        jsStockCategoryList = """
            categoryList = [];
            document.querySelectorAll("div#allSector td.e2 a").forEach(function(category, ci, ca){
                categoryList.push({
                    "name": category.innerText,
                    "url": category.href
                });
            });
        
            return JSON.stringify(categoryList);
            """
        stockCategoryList=json.loads(driver.execute_script(jsStockCategoryList))
        if sectionLi.text=="板块综合":
            #stockSectionCategoryList['shared'] = stockCategoryList#Steve comment on 2021-04-03
            fLog.writeLog(f'No need to capture shared data.')
        elif sectionLi.text=="行业分类":
            stockSectionCategoryList['industry'] = stockCategoryList
        elif sectionLi.text=="地域板块":
            stockSectionCategoryList['region'] = stockCategoryList
        elif sectionLi.text=="概念板块":
            stockSectionCategoryList['concept'] = stockCategoryList
        
    for key in stockSectionCategoryList.keys():
        cnt = len(stockSectionCategoryList[key])
        fLog.writeLog(f"{key} - {cnt}")

    for key in stockSectionCategoryList.keys():
        #key = 'region'
        cnt = len(stockSectionCategoryList[key])
        fLog.writeLog(f"Start to capture section {key} - {cnt}")
        for category in stockSectionCategoryList[key]:
            #industry = stockSectionCategoryList['industry'][0]
            try:
                driver.get(category['url']) 
            except:
                fLog.writeLog(f"Loading page timeout, company is {category['name']}.")
            
            jsStockList = """
                stockList = [];
                document.querySelectorAll("td.e2 a").forEach(function(s, i, a){
                    s1 = s.getAttribute("href")
                    
                    stockList.push(s1.slice(s1.indexOf("/", 1)+1, s1.lastIndexOf("/")));
                });
                return JSON.stringify(stockList);
                """
            stockList=json.loads(driver.execute_script(jsStockList))
            category['stockList'] = stockList

    currentDate = time.strftime("%Y%m%d", time.localtime())

    #Save to file
    dataFile = f"{savePath}stockSectionCategoryList{currentDate}.js"
    f=open(dataFile, 'w')
    json.dump(stockSectionCategoryList, f, ensure_ascii=False) 

    f.flush()
    f.close()

def updateSohuSecSectionCategory(dataFile):
    #Load from file, and update to DB
    f1 = open(dataFile, 'r')
    s1 = json.load(f1)
    f1.close()

    for sectionName in s1.keys():
        #sectionName='region'
        for category in s1[sectionName]:
            otherFields = "('"+sectionName+"','"+category['name']+"','"
            values = otherFields + ("'),"+otherFields).join(category['stockList']) + "')"
            sqlInsertSection = f"""
                INSERT INTO sec_stock_section_pivot(section_name, section_value, stock_code)
                VALUES {values}
                """
            rowcount = cs1.execute(sqlInsertSection)
            fLog.writeLog(f"Stock section {sectionName} - {category['name']} insert {rowcount}.")
            conn.commit()

if __name__ == "__main__":
    dataFile = f"{savePath}secTypeList{currentDate}.js"
    #captureFenghuangSecSectionCategory(dataFile)
    updateFenghuangSecSectionCategory(dataFile)
    
    dataFile = f"{savePath}stockSectionCategoryList{currentDate}.js"
    #captureSohuSecSectionCategory(dataFile)
    #updateSohuSecSectionCategory(dataFile)
    
    cs1.close()
    conn.close()
    driver.close()
    fLog.close()
    
