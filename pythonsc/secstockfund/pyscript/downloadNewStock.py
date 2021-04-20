#Download new stock info
from fhsechead import *
import execjs

currentDate = time.strftime("%Y%m%d", time.localtime())
savePath="C:/fh/testenv1/sec/"

logFile='C:/fh/testenv1/sec/downloadNewStock.log'
fLog = LogWriter(logFile)

driver = webdriver.Chrome(executable_path=chromeDriverExe, chrome_options=chrome_options)
driver.set_page_load_timeout(40)

batchSize = 100

def captureNewStocks():
    fLog.writeLog(f"Start to capture new stocks...")
    try:
        driver.get("http://quote.eastmoney.com/center/gridlist.html#newshares")
    except:
        fLog.writeLog(f'Loading page timeout.')
    """
        document.querySelector("span.paginate_page a.paginate_button:last-child")
        document.querySelector("input.paginate_input")
        document.querySelector("a.paginte_go")
    """
    maxPage= int(driver.find_element(By.CSS_SELECTOR,"span.paginate_page a.paginate_button:last-child").text)
    
    newStockList=[]
    for pagei in range(1, maxPage+1):
    #for pagei in range(1, 5+1):
        pageInput = driver.find_element(By.CSS_SELECTOR,"input.paginate_input")
        goButton= driver.find_element(By.CSS_SELECTOR,"a.paginte_go")
        pageInput.clear()
        pageInput.send_keys(pagei)
        goButton.click()
        time.sleep(1)
        
        jsNewStocks = """
            var stocks = document.querySelectorAll("table#table_wrapper-table tbody tr td:nth-child(3) a")
            var goMarketDates = document.querySelectorAll("table#table_wrapper-table tbody tr td:nth-child(18)")
            var stockList = []
            for(let i=0; i<stocks.length; i++){
                stockName = stocks[i].innerText;
                stockInfo = stocks[i].href.substring(stocks[i].href.lastIndexOf("/")+1).split(".");
                stockList.push({
                    stockCode: stockInfo[1],
                    stockName: stockName,
                    sectionName: 'A股',
                    subsectionName: stockInfo[0]==1 ? '沪A' : (stockInfo[0]==0 ? '深A' : stockInfo[0]),
                    stockType: '股票',
                    shSzIndicator: stockInfo[0]==1 ? 0 : (stockInfo[0]==0 ? 1 : stockInfo[0]),
                    goMarketDate: goMarketDates[i].innerText
                });
            }
            return JSON.stringify(stockList);
            """
        newStockList+=json.loads(driver.execute_script(jsNewStocks))
        
    fLog.writeLog(f"Capture new stock completed, captured {len(newStockList)} stocks.")
    return newStockList
   
def supplementNewStock(stockList):
    for s in stockList:
        sqlCheckExist=f"""
            SELECT 1 
            FROM sec_stock 
            WHERE region_code='CN' AND stock_type='股票' AND stock_code='{s['stockCode']}'
            """
        rowcount = cs1.execute(sqlCheckExist)
        if rowcount == 0:
            sqlInsert=f"""
                INSERT INTO sec_stock(
                    stock_code, stock_name, region_code, section_name, subsection_name, 
                    stock_type, sh_sz_indicator, go_market_date)
                VALUES('{s['stockCode']}','{s['stockName']}', 'CN', '{s['sectionName']}', '{s['subsectionName']}', 
                    '{s['stockType']}', '{s['shSzIndicator']}', '{s['goMarketDate']}')
                """
            rowcount = cs1.execute(sqlInsert)
            fLog.writeLog(f"Saved new Stock {s['stockCode']}-{s['stockName']} into DB.")
        else:
            fLog.writeLog(f"Stock {s['stockCode']}-{s['stockName']} already exists.")
    conn.commit()

if __name__=="__main__":
    newStockList = captureNewStocks()
    supplementNewStock(newStockList)
    
    cs1.close()
    conn.close()
    driver.close()
    fLog.close()
