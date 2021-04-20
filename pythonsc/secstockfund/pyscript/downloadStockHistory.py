from fhsechead import *
from dbpoolutils import *

import multiprocessing

logFile='C:/fh/testenv1/sec/stock-history.log'
fLog = LogWriter(logFile)

currentDate = time.strftime("%Y%m%d", time.localtime())
savePath = "C:/fh/testenv1/sec/stockhistory"

isSaveData2CsvFile = False
#To reduce IO and improve performance, also protect hard disk, so normally set False

def downloadStockHistory(stockCode, isSaveData2CsvFile):
    #stockCode = "0600756" #Sample
    #stockCode = "1200168"
    #stockCode = "0605199" #葫芦娃
    #url = f"http://quotes.money.163.com/service/chddata.html?code={stockCode}&start=20200101&end=20200131&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;VOTURNOVER;"
    url = f"http://quotes.money.163.com/service/chddata.html?code={stockCode}"
    
    srcDataFile = f"{savePath}/{stockCode}-{currentDate}.csv"
    content = None
    try:
        res = requests.get(url, headers=headers)
        content = res.content.decode('gbk') #res.text()采用默认的编码方式，有些会中文乱码，因此需读字节并指定解码规则
        
        if isSaveData2CsvFile==False:
            return 0, content
        else:
            f=open(srcDataFile, 'w', newline='') #newline参数是为了解决\r\n问题，否则每行后都会有一空行
            
            f.write(content)
            f.flush()
            f.close()
            
            return 0, 'dummy'
    except Exception as e:
        fLog.writeLog(f"Call downloadStockHistory() failed, stock code={stockCode} exception={e}")
        return -1, 'dummy'

def stockCallback(returnCode):
    fLog.writeLog(f"Call downloadStockHistory() finished status={returnCode}")
    
def loadStockHistory2DB(stockCode, rows):
    connx = None
    csx = None
    try:
        connx = dbPool.connection()
        csx=connx.cursor()
        
        sqlMaxPriceDate = f"""
            SELECT DATE_FORMAT(MAX(price_date), '%Y-%m-%d') max_price_date
            FROM sec_stock_history 
            WHERE price_date>=DATE_SUB(CURDATE(), INTERVAL {g_interval_days} DAY) AND stock_code='{stockCode}'
            """
        rowcount = csx.execute(sqlMaxPriceDate)
        maxPriceDate = csx.fetchall()[0][0] #((None,),)
        maxPriceDate = '2010-01-01' if maxPriceDate==None else maxPriceDate
        
        for r in rows:
            if r[0] > maxPriceDate:  #'>' not supported between instances of 'str' and 'NoneType'
                sqlInsertHistory = f"""
                    INSERT INTO sec_stock_history(
                        price_date,stock_code,stock_name,
                        close_price,high_price,low_price,open_price,yest_close,
                        delta_amount,delta_percent,handover_rate,trading_volume,turnover_amount,
                        total_market_value,circu_market_value,trading_lots)
                    VALUES('{r[0]}', '{stockCode}', '{r[2]}',
                        {'null' if(r[3]=='None' or r[3]=='') else r[3]}, 
                        {'null' if(r[4]=='None' or r[4]=='') else r[4]}, 
                        {'null' if(r[5]=='None' or r[5]=='') else r[5]}, 
                        {'null' if(r[6]=='None' or r[6]=='') else r[6]}, 
                        {'null' if(r[7]=='None' or r[7]=='') else r[7]}, 
                        
                        {'null' if(r[8]=='None' or r[8]=='') else r[8]}, 
                        {'null' if(r[9]=='None' or r[9]=='') else r[9]}, 
                        {'null' if(r[10]=='None' or r[10]=='') else r[10]}, 
                        {'null' if(r[11]=='None' or r[11]=='') else r[11]}, 
                        {'null' if(r[12]=='None' or r[12]=='') else r[12]}, 
                        
                        {'null' if(r[13]=='None' or r[13]=='') else r[13]}, 
                        {'null' if(r[14]=='None' or r[14]=='') else r[14]}, 
                        {'null' if(r[15]=='None' or r[15]=='') else r[15]})
                    """
                rowcount = csx.execute(sqlInsertHistory)
            else:
                fLog.writeLog(f"Break for loop at {r[0]} - {maxPriceDate}")
                break;
        connx.commit()
        
        csx.close()
        connx.close()
        fLog.writeLog(f"Call loadStockHistory2DB() finished for stock {stockCode}-{rows[1][2]}.")
    except Exception as e:
        if(csx!=None):
            csx.close()
        if(connx!=None):
            connx.close()
        fLog.writeLog(f"Call loadStockHistory2DB() exception for stock {stockCode}!!!")
        fLog.writeLog(e)
    
def loadStockHistory2DBMain(stockCode, forceDownload=False):
    #Check data file exists
    srcDataFile = f"{savePath}/{stockCode}-{currentDate}.csv"
    fLog.writeLog(f"loadStockHistory2DB - srcDataFile: {srcDataFile}")
    try:
        #if not, then download; or force download.
        returnCode=0
        checkFile = os.path.isfile(srcDataFile)
        if checkFile==False or forceDownload==True: #If file not exist, then download.
            fLog.writeLog(f"Start download stock info {stockCode} ...")
            returnCode, content = downloadStockHistory(stockCode, isSaveData2CsvFile)
            fLog.writeLog(f"Download stock info {stockCode} {'success.' if returnCode==0 else 'failed!'}")
        
            if returnCode==-1:
                fLog.writeLog(f"Srouce file not exists, and download stock info failed. stockCode={stockCode}")
                return -1;
                
        rows=[]
        if returnCode==0:
            if isSaveData2CsvFile==False:
                rows = list(csv.reader(io.StringIO(content)))
            else:
                #If file exists, then use it
                #if yes, then load
                with open(srcDataFile, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
        
        if len(rows)>=2:
            loadStockHistory2DB(stockCode, rows[1:])
        else:
            fLog.writeLog(f"No data in source file for stock {stockCode}.")
        
        return 0
    except Exception as e:
        fLog.writeLog(f"Call loadStockHistory2DBMain() failed, exception={e}")
        return -1
    
if __name__ == "__main__":
    startTime = datetime.datetime.now()
    sqlStockList = """
        SELECT CONCAT(sh_sz_indicator, stock_code) stock_code, stock_name 
        FROM sec_stock a
        WHERE sh_sz_indicator IS NOT NULL 
            AND NOT EXISTS(
                SELECT 1 FROM (
			SELECT DISTINCT stock_code FROM sec_stock_history 
			WHERE price_date>=DATE_SUB(CURDATE(), INTERVAL 1 DAY)
		) b
                WHERE CONCAT(a.sh_sz_indicator, a.stock_code)=b.stock_code
            )
        ORDER BY 1
        """
        #AND stock_name like '%酒%'
        #AND stock_type='指数'
        #AND stock_type='股票'
        #AND stock_name like '%茅台%'
    #Download stock history only for big up stocks, this can improve performance.
    sqlStockListx = """
        SELECT CONCAT(sh_sz_indicator, stock_code) stock_code, stock_name 
        FROM sec_stock a
        WHERE price_time>='2021-04-12' AND percent >=0.09 
        	AND CONCAT(sh_sz_indicator, stock_code) NOT IN (
        	SELECT stock_code FROM sec_stock_history WHERE price_date>=DATE_SUB(CURDATE(), INTERVAL 10 DAY)
        	)
        ORDER BY 1
        """
        #
    sqlStockListx = """
        SELECT stock_code, stock_name FROM sec_stock_history WHERE price_date='2021-04-09' AND delta_percent>=9
        UNION
        SELECT CONCAT(sh_sz_indicator, stock_code) stock_code, stock_name FROM sec_stock WHERE price_time>='2021-04-12' AND percent >=0.09
        ORDER BY stock_code
        """
        
    #Testing
    sqlStockListx = """
        SELECT CONCAT(sh_sz_indicator, stock_code) stock_code, stock_name 
        FROM sec_stock WHERE stock_code='600519'
        """
    rowcount = cs1.execute(sqlStockList)
    conn.commit()
    stocks = cs1.fetchall()
    pool = multiprocessing.Pool(processes=5)
    total = len(stocks)
    i = 0
    for s in stocks:
        i = i + 1
        fLog.writeLog(f"Start load stock into DB {i}/{total} - {s[0]} - {s[1]}")
        try:
            #Way 1: single process
            #returnCode = loadFund2DB(f[0])
            #fLog.writeLog(f"Load stock into DB {s[0]} {'success.' if returnCode==0 else 'failed!'}")
            
            #Way 2: multiprocessing
            pool.apply_async(loadStockHistory2DBMain, args=(s[0], ), callback=stockCallback)
            
            #Dedug under single process
            #loadStockHistory2DBMain(s[0])
            #print(s[0])
        except Exception as e:
            fLog.writeLog(f"Load stock into DB {s[0]} interrupt with exception {e}.")
    pool.close()
    pool.join()
    fLog.writeLog(f"All processes are completed, multiprocessing pool closed.")

    endTime = datetime.datetime.now()
    fLog.writeLog(f"Download {rowcount} stocks history total spent time {(endTime-startTime).seconds}s")

    cs1.close()
    conn.close()
    dbPool.close()
    fLog.writeLog(f"All processes are completed, DB connection and DB pool connections are closed.")
    fLog.close()
    
    
"""
网易的接口是：
网易的数据格式为csv文件

日线URL格式：http://quotes.money.163.com/service/chddata.html?code=代码&start=开始时间&end=结束时间&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP

参数说明：代码为股票代码，上海股票前加0，如600756变成0600756，深圳股票前加1。时间都是6位标识法，年月日，fields标识想要请求的数据。可以不变。

例如大盘指数数据查询（上证指数000001前加0，沪深300指数000300股票前加0，深证成指399001前加1，中小板指399005前加1，创业板指399006前加1）： 
http://quotes.money.163.com/service/chddata.html?code=0000300&start=20151219&end=20171108&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;VOTURNOVER

上海股票数据查询（浪潮）：http://quotes.money.163.com/service/chddata.html?code=0600756&start=20160902&end=20171108&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;VOTURNOVER;
"""
