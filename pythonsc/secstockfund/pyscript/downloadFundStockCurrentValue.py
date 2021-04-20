#Capture stock current data and update into DB
from fhsechead import *
from dbpoolutils import *

import multiprocessing

logFile='C:/fh/testenv1/sec/fund-updatefundtrend.log'
fLog = LogWriter(logFile)

currentDate = time.strftime("%Y%m%d", time.localtime())

def downloadStockCurrent(stockCodeList):
    stockCodeListStr = ",".join(stockCodeList)
    #stockCodeList = "0600519,0601318,1000333,0600036" #Sample
    #url = f"http://api.money.126.net/data/feed/0600519,0601318,1000333,0600036,money.api"
    url = f"http://api.money.126.net/data/feed/{stockCodeListStr},money.api"
    
    connx = None
    csx = None
    
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
                    WHERE stock_code='{stockCode[1:]}' AND sh_sz_indicator='{stockCode[:1]}'
                    """
                connx = dbPool.connection()
                csx=connx.cursor()
                rowcount = csx.execute(sqlUpdatePrice)
                connx.commit()
                
                csx.close()
                connx.close()
            else:
                fLog.writeLog(f"No data - {stock}")
        conn.commit()
    except Exception as e:
        if(csx!=None):
            csx.close()
        if(connx!=None):
            connx.close()
        fLog.writeLog(f"Call downloadStockCurrent() exception for stock={stockCodeListStr}, exception={e}")
        return -1
    
    return 0

def stockCallback(status):
    fLog.writeLog(f"Call downloadStockCurrent() finished status={status}")
    
def stockCurrentMain():
    startTime = datetime.datetime.now()
    sqlUpdateShSzIndicator = """
        UPDATE sec_stock
        SET sh_sz_indicator = CASE 
            WHEN stock_type='指数' AND section_name='上证' THEN 0 
            WHEN stock_type='指数' AND section_name='深证' THEN 1 
            WHEN stock_type='股票' AND SUBSTRING(stock_code, 1, 1) IN ('0','2','3') THEN 1 
            ELSE 0 END
        WHERE region_code='CN' AND section_name<>'中证'
        """
    rowcount = cs1.execute(sqlUpdateShSzIndicator)
    conn.commit()
    fLog.writeLog(f"Updated stock SH_SZ_INDICATOR {rowcount} records.")
    
    sqlStockCodeList = """
        SELECT CONCAT(sh_sz_indicator, stock_code) stock_code 
        FROM sec_stock t 
        WHERE region_code='CN' AND section_name<>'中证' 
        ORDER BY fund_welcome_cnt DESC
        """
        # 股票代码前加前缀：上证加0, 深证加1，中证暂未知
        #AND price_time IS NULL
        
    #只快速更新前两天在大涨的股票，用于快速跟踪大涨股
    sqlStockCodeListx = """
        SELECT CONCAT(sh_sz_indicator, stock_code) stock_code 
        FROM sec_stock t 
        WHERE region_code='CN' AND section_name<>'中证' 
		AND EXISTS(
			SELECT 1 
			FROM sec_stock_continuity c 
			WHERE end_date>=DATE_SUB(CURDATE(), INTERVAL 4 DAY) 
				AND c.up9_days >=2
				AND c.stock_code = CONCAT(t.sh_sz_indicator, t.stock_code)
		)
        ORDER BY fund_welcome_cnt DESC
        """
    
    rowcount = cs1.execute(sqlStockCodeList)
    records = cs1.fetchall()
    stockCodeList = []
    for r in records:
        stockCodeList.append(r[0])
    
    pool = multiprocessing.Pool(processes=5)
    batchUnit = 500
    batchs = math.ceil(len(stockCodeList)/batchUnit)
    for i in range(batchs):
        fLog.writeLog(f"Update stock price for {0+i*batchUnit} - {batchUnit*(i+1)}:")
        #Way 1: single process
        #status = downloadStockCurrent(stockCodeList[0+i*batchUnit:batchUnit*(i+1)])
        #fLog.writeLog(f"Call downloadStockCurrent() status={status}")
        #Way 2: multiprocessing
        pool.apply_async(downloadStockCurrent, args=(stockCodeList[0+i*batchUnit:batchUnit*(i+1)], ), callback=stockCallback)
    
    pool.close()
    pool.join()
    endTime = datetime.datetime.now()
    fLog.writeLog(f"Update {batchUnit} batch unit * {batchs} batchs = {rowcount} stocks current value total spent time {(endTime-startTime).seconds}s")


#Capture fund current data and update into DB
def downloadFundCurrent(fundCode):
    #fundCode = "161725" #Sample
    fLog.writeLog(f"Call downloadFundCurrent() start for fund={fundCode}...")
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
        connx = dbPool.connection()
        csx=connx.cursor()
        rowcount = csx.execute(sqlUpdatePrice)
        connx.commit()
        
        csx.close()
        connx.close()
        fLog.writeLog(f"Call downloadStockCurrent() finished for fund={fundCode}.")
    except Exception as e:
        fLog.writeLog(f"Call downloadStockCurrent() exception for fund={fundCode}, exception={e}.")
        #fLog.writeLog(e)
        return -1
    
    return 0
    
def fundCallback(status):
    fLog.writeLog(f"Call downloadFundCurrent() finished status={status}")
    
def fundCurrentMain():
    startTime = datetime.datetime.now()
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
        ) t WHERE rownum<=1000
        """
    rowcount = cs1.execute(sqlFundCodeList)
    records = cs1.fetchall()
    fundCodeList = []
    for r in records:
        fundCodeList.append(r[0])
    
    pool = multiprocessing.Pool(processes=5)
    for fundCode in fundCodeList:
        #Way 1: single process
        #status = downloadFundCurrent(fundCode)
        #Way 2: multiprocessing
        status = pool.apply_async(downloadFundCurrent, args=(fundCode, ), callback=fundCallback)
        fLog.writeLog(f"Call downloadStockCurrent() for fund={fundCode} status={status}")
    pool.close()
    pool.join()
    fLog.writeLog(f"All processes for updating fund current value are completed, multiprocessing pool closed.")
    
    endTime = datetime.datetime.now()
    fLog.writeLog(f"Update {rowcount} funds current value total spent time {(endTime-startTime).seconds}s")

if __name__ == "__main__":
    stockCurrentMain()
    #fundCurrentMain()
    
    cs1.close()
    conn.close()
    dbPool.close()
    fLog.close()
