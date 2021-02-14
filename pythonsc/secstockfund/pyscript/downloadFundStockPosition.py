#Update fund stock position shares
from fhsechead import *

logFile='C:/fh/testenv1/sec/fund.log'
fLog=open(logFile, 'a+')

def writeLog(msg):
    msgx = f'\n[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] {msg}'
    print(msgx)
    fLog.writelines([msgx])
    fLog.flush()

driver = webdriver.Chrome(executable_path=chromeDriverExe, chrome_options=chrome_options)
driver.set_page_load_timeout(40)

def updateFundPositionShares(fundCodeList):
    #for fundCode in fundCodeList:
    for index, fundCode in enumerate(fundCodeList):
        writeLog(f"Start to update fund position shares {index}/{len(fundCodeList)} - {fundCode}")
        #fundCode = fundCodeList[0]
        try:
            driver.get(f"http://fund.eastmoney.com/{fundCode}.html") # get接受url可以是如何网址，此处以百度为例
        except:
            writeLog(f'Loading page timeout, fund code is {fundCode}.')
        
        jsFundPositionShares = """
            positionShares = []
            stockLinkList = document.querySelectorAll("li#position_shares table tr td:nth-child(1)")
            stockPercentList = document.querySelectorAll("li#position_shares table tr td:nth-child(2)")
            for(let i=0; i<stockLinkList.length; i++){
            	stockLink = stockLinkList[i].children[0].href;
            	stockPercentStr = stockPercentList[i].innerText;
            	
            	stockCode=stockLink.slice(stockLink.lastIndexOf("/")+1+2, stockLink.lastIndexOf("."));
            	percent = parseFloat(stockPercentStr.slice(0, stockPercentStr.length-1));
            	
            	positionShares.push({
            		"stockCode" : stockCode,
            		"stockPercent" : percent
            	});
            }
            
            return JSON.stringify(positionShares);
            """
            
        fundPositionShares=json.loads(driver.execute_script(jsFundPositionShares))
        for ps in fundPositionShares:
            #ps = fundPositionShares[0]
            sqlUpdateFundPositionShares = f"""
                UPDATE sec_fund_stock_mapping
                SET fund_percent={ps['stockPercent']}
                WHERE fund_code='{fundCode}' AND SUBSTR(stock_code,1,6)='{ps["stockCode"]}'
                """
            rowcount = cs1.execute(sqlUpdateFundPositionShares)
        conn.commit()
        writeLog(f"Fund {fundCode} position shares has been updated in DB.")

if __name__ == "__main__":
    sqlFundStockList = f"""
        SELECT fund_code
        FROM sec_fund 
        WHERE fh_mark IS NOT NULL AND fh_mark='+'
        """
    rowcount = cs1.execute(sqlFundStockList)
    result = cs1.fetchall()
    fundCodeList = []
    for s in result:
        fundCodeList.append(s[0])
        
    updateFundPositionShares(fundCodeList)

    cs1.close()
    conn.close()
    driver.close()