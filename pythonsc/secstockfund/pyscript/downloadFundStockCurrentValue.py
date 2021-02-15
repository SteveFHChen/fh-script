#Capture stock current data and update into DB
from fhsechead import *

logFile='C:/fh/testenv1/sec/fund-updatefundtrend.log'
fLog = LogWriter(logFile)

currentDate = time.strftime("%Y%m%d", time.localtime())

def downloadStockCurrent(stockCodeList):
    stockCodeListStr = ",".join(stockCodeList)
    #stockCodeList = "0600519,0601318,1000333,0600036" #Sample
    #url = f"http://api.money.126.net/data/feed/0600519,0601318,1000333,0600036,money.api"
    url = f"http://api.money.126.net/data/feed/{stockCodeListStr},money.api"
    
    try:
        res = requests.get(url, headers=headers)
        content = res.text
        result = json.loads(content[len("_ntes_quote_callback("):-len(");")])
        
        #time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        for stockCode in result.keys():
            #stockCode='1000023'
            stock = result[stockCode]
            if 'yestclose' in stock.keys():
                sqlUpdatePrice = f"""
                    UPDATE sec_stock
                    SET price_time='{stock['time']}', yest_close={stock['yestclose']}, now_price={stock['price']}, arrow='{stock['arrow']}', 
                        open_price={stock['open']}, high_price={stock['high']}, low_price={stock['low']}, volume={stock['volume']}, percent={stock['percent']}
                    WHERE stock_code='{stockCode[1:]}'
                    """
                rowcount = cs1.execute(sqlUpdatePrice)
            else:
                fLog.writeLog(f"No data - {stock}")
        conn.commit()
    except Exception as e:
        fLog.writeLog(f"Exception msg: {e}")
        return -1
    
    return 0

def stockCurrentMain():
    sqlStockCodeList = """
        SELECT CONCAT(CASE WHEN SUBSTRING(stock_code, 1, 1) IN ('0','3') THEN 1 ELSE 0 END, stock_code) stock_code 
        FROM sec_stock t 
        WHERE region_code='CN' AND price_time IS NULL
        ORDER BY fund_welcome_cnt DESC
        """
    rowcount = cs1.execute(sqlStockCodeList)
    records = cs1.fetchall()
    stockCodeList = []
    for r in records:
        stockCodeList.append(r[0])
    batchUnit = 500
    for i in range(math.ceil(len(stockCodeList)/batchUnit)):
        fLog.writeLog(f"Update stock price for {0+i*batchUnit} - {batchUnit*(i+1)}:")
        status = downloadStockCurrent(stockCodeList[0+i*batchUnit:batchUnit*(i+1)])
        fLog.writeLog(f"Call downloadStockCurrent() status={status}")


#Capture fund current data and update into DB
def downloadFundCurrent(fundCode):
    #fundCode = "161725" #Sample
    url = f"http://fundgz.1234567.com.cn/js/{fundCode}.js"
    
    try:
        res = requests.get(url, headers=headers)
        content = res.text
        fund = json.loads(content[len("jsonpgz("):-len(");")])
        
        sqlUpdatePrice = f"""
            UPDATE sec_fund
            SET net_date='{fund['jzrq']}', unit_net={fund['dwjz']},esti_net={fund['gsz']},esti_rate={fund['gszzl']},gztime='{fund['gztime']}'
            WHERE fund_code='{fund['fundcode']}'
            """
        rowcount = cs1.execute(sqlUpdatePrice)
        conn.commit()
    except Exception as e:
        fLog.writeLog(e)
        return -1
    
    return 0

def fundCurrentMain():
    sqlFundCodeList = """
        SELECT fund_code 
        FROM sec_fund
        ORDER BY fund_code DESC
        """
    sqlFundCodeList="""
        SELECT fund_code FROM (
        	SELECT @rownum:=@rownum+1 rownum, a.*
        	FROM sec_fund a, (SELECT @rownum:=0) b
        	WHERE fh_mark IS NOT NULL AND fund_type NOT IN ('ETF-场内', '债券型') 
        	ORDER BY fund_scale DESC
        ) t WHERE rownum<=10
        """
    rowcount = cs1.execute(sqlFundCodeList)
    records = cs1.fetchall()
    fundCodeList = []
    for r in records:
        fundCodeList.append(r[0])
    for fundCode in fundCodeList:
        status = downloadFundCurrent(fundCode)
        fLog.writeLog(f"Call downloadStockCurrent() status={status}")

if __name__ == "__main__":
    stockCurrentMain()
    fundCurrentMain()
    
    cs1.close()
    conn.close()
    fLog.close()
