
from fhsechead import *

logFile='C:/fh/testenv1/sec/stock.log'
fLog=open(logFile, 'a+')

def writeLog(msg):
    msgx = f'\n[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] {msg}'
    print(msgx)
    fLog.writelines([msgx])
    fLog.flush()

#Capture all CN stock list, and save into DB
def captureCNStocks():
    writeLog("Start to capture CN stocks...")
    driver = webdriver.Chrome(executable_path=chromeDriverExe, chrome_options=chrome_options)
    driver.set_page_load_timeout(40)

    try:
        driver.get("http://www.yz21.org/stock/info/") # get接受url可以是如何网址，此处以百度为例
    except:
        writeLog(f'Loading page timeout.')

    jsPageNumber = """
        pages = document.querySelectorAll("div.pagestock a")
        return parseInt(pages[pages.length-1].innerText)
        """
    pageNumber = driver.execute_script(jsPageNumber)

    jsCatCNStockList = """
        stockList = []
        stockRows = document.querySelectorAll("table.stockBlock tr")
        stockRows.forEach(function(s){
            if(s.childNodes[2].innerText!="股票代码")
                stockList.push({
                    "code": s.childNodes[2].innerText,
                    "name": s.childNodes[3].innerText,
                    "company": s.childNodes[4].innerText,
                    "shortPinYin": s.childNodes[5].innerText
                });
        });
        return JSON.stringify(stockList);
        """
        
    for i in range(2, pageNumber+1):
    #for i in range(2, 4+1): #For testing
        url = f"http://www.yz21.org/stock/info/stocklist_{i}.html"
        writeLog(f"Start capture page [{i}] {url}")
        try:
            driver.get(url)
        except:
            writeLog(f'Loading page timeout.')
        
        cnStockList=json.loads(driver.execute_script(jsCatCNStockList))
        for s in cnStockList:
            sqlKeyCheck = f"""
                SELECT 
                    stock_code 
                FROM sec_stock 
                WHERE stock_code = '{s['code']}'
                """
            sqlAdd = f"""
                INSERT INTO sec_stock(stock_code, stock_name, short_pinyin, company_name, region_code)
                VALUES ('{s['code']}', '{s['name']}', '{s['shortPinYin']}', '{s['company']}', 'CN')
                """
            try:
                rowcount = cs1.execute(sqlKeyCheck)
                if rowcount>0:
                    writeLog(f"Stock already exists {s['code']} - {s['name']}")
                else:
                    rowcount = cs1.execute(sqlAdd)
            except Exception as e:
                writeLog(f"Failed add: {s}")
                writeLog(f"Exception msg: {e}")
        conn.commit()
        writeLog(f"Completed capture page [{i}].")

    driver.close()
    writeLog("Finished capture CN stocks.")

#Capture all US stock list, and save into DB
def captureUSStocks():
    driver = webdriver.Chrome(executable_path=chromeDriverExe)
    driver.set_page_load_timeout(40)

    try:
        driver.get("http://quote.eastmoney.com/usstocklist.html") # get接受url可以是如何网址，此处以百度为例
    except:
        writeLog(f'Loading page timeout.')

    jsCatUSStockList = """
        usStockaList = document.querySelectorAll("div#quotesearch ul li a")
        usStockList = []
        for(let i=0; i<usStockaList.length; i++){
            usStockList.push({
                'code': usStockaList[i].getAttribute('href').slice('http://quote.eastmoney.com/us/'.length, -".html".length),
                'title': usStockaList[i].getAttribute('title')
            })
        }
        return JSON.stringify(usStockList);
        """
    usStockList=json.loads(driver.execute_script(jsCatUSStockList))

    for s in usStockList:
        titlei = s['title'].replace('\'', '\'\'')
        sqlKeyCheck = f"""
            SELECT 
                stock_code 
            FROM sec_stock 
            WHERE stock_code = '{s['code']}'
            """
        sqlAdd = f"""
            INSERT INTO sec_stock(stock_code, stock_name, region_code)
            VALUES ('{s['code']}', '{titlei}', 'US')
            """
        try:
            rowcount = cs1.execute(sqlKeyCheck)
            if rowcount>0:
                writeLog(f"Stock already exists {s['code']} - {s['title']}")
            else:
                rowcount = cs1.execute(sqlAdd)
        except Exception as e:
            writeLog("Failed add: {s}")
    conn.commit()
    driver.close()

#Capture HK all stocks, and save into DB
def captureHKStocks():
    driver = webdriver.Chrome(executable_path=chromeDriverExe)
    driver.set_page_load_timeout(40)

    try:
        driver.get("http://app.finance.ifeng.com/hq/list.php?type=hkstock") # get接受url可以是如何网址，此处以百度为例
    except:
        writeLog(f'Loading page timeout.')

    jsCatHKStockList = """
        stockList = []
        document.querySelectorAll("div.result ul li a").forEach(function(s, i, a){
            title = s.innerText;
            startIndex = title.lastIndexOf("(")
            endIndex = title.lastIndexOf(")")
            code = title.slice(startIndex+1, endIndex)
            name = title.slice(0, startIndex)
            stockList.push({
                "code": code,
                "name": name
            });
        });
        return JSON.stringify(stockList);
        """
    hkStockList=json.loads(driver.execute_script(jsCatHKStockList))

    for s in hkStockList:
        name = s['name'].replace('\'', '\'\'')
        sqlKeyCheck = f"""
            SELECT 
                stock_code 
            FROM sec_stock 
            WHERE stock_code = '{s['name']}'
            """
        sqlAdd = f"""
            INSERT INTO sec_stock(stock_code, stock_name, region_code)
            VALUES ('{s['code']}', '{name}', 'HK')
            """
        try:
            rowcount = cs1.execute(sqlKeyCheck)
            if rowcount>0:
                writeLog(f"Stock already exists {s['code']} - {s['title']}")
            else:
                rowcount = cs1.execute(sqlAdd)
        except Exception as e:
            writeLog("Failed add: {s}")
    conn.commit()
    driver.close()

#Base on fund_stock_mapping to find missing stock, and supplement to DB
def downloadMissingStock(stockList, stockList2i):
    #stockList2i = stockList2[0+i*batchUnit:batchUnit*(i+1)]
    stockList2Str = ",".join(stockList2i)
    url = f"http://hq.sinajs.cn/list={stockList2Str}"
    res = requests.get(url, headers=headers)
    content = res.text
    for index, line in enumerate(content.split("\n")[:-1]):
        stockInfoStr = js2py.eval_js(line)
        writeLog(f"{stockList[index]} - {stockInfoStr}")
        stockInfo = stockInfoStr.split(",")
        if len(stockInfo)>=6:
            sqlAddStock = f"""
                INSERT INTO sec_stock(stock_code, stock_name, region_code)
                VALUES('{stockList[index]}', '{stockInfo[0].replace(" ", "")}', 'CN')
                """
            #print(sqlAddStock)
            rowcount = cs1.execute(sqlAddStock)
    conn.commit()
    """
    jsStock = execjs.compile(content)
    for s2 in stockList2i:
        stockInfo = jsStock.eval(f"hq_str_{s2}")#中文乱码
        print(stockInfo)
    """
    return 0
    
def downloadMissingStockMain():
    sqlMissingStock = """
        SELECT stock_code, 
            CASE 
            WHEN SUBSTR(region_stock_code,7)='1' THEN CONCAT('s_sh',stock_code)
            WHEN SUBSTR(region_stock_code,7)='2' THEN CONCAT('s_sz',stock_code)
            END AS stock_code2
        FROM(
            SELECT DISTINCT m.stock_code, m.region_stock_code
            FROM sec_fund_stock_mapping m 
            LEFT JOIN sec_stock s ON m.stock_code=s.`stock_code`
            WHERE s.stock_code IS NULL AND LENGTH(m.region_stock_code)=7 AND SUBSTR(region_stock_code,7) IN ('1', '2')
        ) t WHERE stock_code IS NOT NULL 
        """
    rowcount = cs1.execute(sqlMissingStock)
    result = cs1.fetchall()
    stockList = []
    stockList2 = []
    for r in result:
        stockList.append(r[0])
        stockList2.append(r[1])
        
    batchUnit = 10
    for i in range(math.ceil(len(stockList2)/batchUnit)):
        writeLog(f"Download missing stocks for {0+i*batchUnit} - {batchUnit*(i+1)}:")
        status = downloadMissingStock(
            stockList[0+i*batchUnit:batchUnit*(i+1)], 
            stockList2[0+i*batchUnit:batchUnit*(i+1)])
        writeLog(f"Call downloadMissingStock() status={status}")

if __name__ == "__main__":
    captureCNStocks()
    #downloadMissingStockMain()

    cs1.close()
    conn.close()
    fLog.close()