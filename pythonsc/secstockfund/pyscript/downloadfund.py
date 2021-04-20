
#多线程、多进程下载更新基金净值
#多线程、多进程计算数据，以提高图像效率

"""
    python 多进程pool.apply_async 在报错 AttributeError: Can't get attribute
    https://blog.csdn.net/alanguoo/article/details/81461298
    
    ModuleNotFoundError: No module named 'DBUtils'
    https://www.cnblogs.com/fadedlemon/p/13794749.html
    
    Mysql系统参数查询和设置
    https://blog.csdn.net/qq_35001776/article/details/83072400
    
    Python数据库连接池实例——PooledDB
    https://www.cnblogs.com/Xjng/p/3437694.html
    
"""

#下载所有基金的详细信息
"""
1) Check the file whether exists in local
    if yes, go to step 2
    if no, then download from internet as original data
2) Use the file
    read the file, and load the data into DB.
"""
from fhsechead import *
from dbpoolutils import *

import multiprocessing
import execjs

currentDate = time.strftime("%Y%m%d", time.localtime())
#currentDate = '20210130'
savePath = "C:/fh/testenv1/sec/cat20210130"
#savePath = "C:/fh/testenv1/sec/cat20210210"

logFile='C:/fh/testenv1/sec/fund-updatefundtrend.log'
fLog = LogWriter(logFile)

#Using approach 1
def downloadFund(fundCode):
    url = f"http://fund.eastmoney.com/pingzhongdata/{fundCode}.js"
    #http://fund.eastmoney.com/pingzhongdata/161725.js
    srcDataFile = f"{savePath}/{fundCode}-{currentDate}.js"
    try:
        res = requests.get(url, headers=headers)
        content = res.text
        
        #time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        f=open(srcDataFile, 'a+')
        
        f.writelines([content])
        f.flush()
        f.close()
    except:
        return -1, srcDataFile
    
    return 0, srcDataFile

def loadNetWorthTrend(jsContent):
    
    rstcode = -1
    fundCode = jsContent.eval("fS_code")
    netWorthTrend = jsContent.eval("Data_netWorthTrend")
    acWorthTreand = jsContent.eval("Data_ACWorthTrend")
    
    fLog.writeLog(f"Start to load net worth trend - fund code: {fundCode}")
    
    #Performance tunned, fast get max net date, and fast add not existing into DB
    sqlMaxNetDate = f"""
        SELECT fund_code, MAX(net_date) max_net_date 
        FROM sec_fund_trend 
        WHERE net_date >=UNIX_TIMESTAMP(DATE_SUB(CURDATE(), INTERVAL {g_interval_days} DAY))*1000 AND fund_code='{fundCode}'
        """
    connx = dbPool.connection()
    csx=connx.cursor()
    rowcount = csx.execute(sqlMaxNetDate)
    result = csx.fetchall()
    maxNetDate = result[0][1]
    if maxNetDate==None:
        fLog.writeLog(f"Exception: didn't get maxNetDate, please change interval day to larger. - fund code: {fundCode}")
    
    if len(netWorthTrend)==len(acWorthTreand):
        #for i in range(len(netWorthTrend)): #This is for inital full data loading
        for i in np.arange(1, 1000)*(-1): #For performance tuning, 取倒数n个最近的
            ni = netWorthTrend[i]
            if ni['x'] > maxNetDate:
                #print(f"insert i={i}")
                ai = acWorthTreand[i]
                sql = f"""
                        INSERT INTO sec_fund_trend(net_date, fund_code, unit_net, acc_net, day_delta_rate)
                        VALUES ({ni['x']}, '{fundCode}', {ni['y']}, {ai[1]}, {ni['equityReturn']})
                    """
                rowcount = csx.execute(sql)
            else:
                fLog.writeLog(f"break i={i}, fundCode={fundCode}")
                break;
        connx.commit()
        rstcode = 0
    
    csx.close()
    connx.close()
    
    return rstcode;
    
def analyzeFundLastNRate(fundCode):
    fLog.writeLog(f"Start to analyze fund last N rate - fund code: {fundCode}")
    #fundCode = '511880'
    sqlOriRate = f"""
        SELECT net_date, day_delta_rate
        FROM sec_fund_trend 
        WHERE net_date>=UNIX_TIMESTAMP('2021-01-01')*1000 AND fund_code='{fundCode}'
        ORDER BY net_date
        """
    connx = pool.connection()
    csx=connx.cursor()
    rowcount = csx.execute(sqlOriRate)
    result = csx.fetchall()
    connx.commit()
    
    accRates = []
    accRate = 0
    for r in result:
        accRate+=r[1]
        accRates.append(round(accRate,4))
    maxRate, minRate = max(accRates), min(accRates)
    maxIndex, minIndex = accRates.index(maxRate), accRates.index(minRate)
    maxDate, minDate = result[maxIndex][0], result[minIndex][0]
    nowRate, nowDate = accRates[-1], result[-1][0]
    sqlSaveAnalyze = f"""
        UPDATE sec_fund
        SET lastn_top_rate={maxRate}, lastn_top_date=FROM_UNIXTIME({maxDate}/1000),
            lastn_low_rate={minRate}, lastn_low_date=FROM_UNIXTIME({minDate}/1000),
            last_now_rate ={nowRate}, last_now_date =FROM_UNIXTIME({nowDate}/1000)
        WHERE fund_code='{fundCode}'
        """
    rowcount = csx.execute(sqlSaveAnalyze)
    connx.commit()
    csx.close()
    connx.close()
    
    return 0
    
def loadFundStockMapping(jsContent):
    fundCode = jsContent.eval("fS_code")
    stockCodes = jsContent.eval("stockCodes")
    
    fLog.writeLog(f"Start to load fund stock mapping - fund code: {fundCode}")
    
    #To improve performance, use cs1.executemany() to batch insert records, instead each records execute a sql.
    mappings = []
    for s in stockCodes:
        stockCode = s[:6] if s.isnumeric() and len(s)==7 else s
        mappings+=[(fundCode, s, stockCode)]
        '''
        sql = f"""
            INSERT INTO sec_fund_stock_mapping(fund_code, region_stock_code, stock_code)
            VALUES('{fundCode}', '{s}', '{stockCode}');
            """
        #print(sql)
        rowcount = cs1.execute(sql)
        '''
    sql = f"""
        INSERT INTO sec_fund_stock_mapping(fund_code, region_stock_code, stock_code)
        VALUES(%s, %s, %s);
        """
    cs1.executemany(sql, mappings)
    conn.commit()
    return 0

def updateFundScale(jsContent):
    try:
        fundCode = jsContent.eval("fS_code")
        fundScale = jsContent.eval("Data_fluctuationScale")
        latestFundScale = fundScale['series'][-1]['y']
        
        fLog.writeLog(f"Start to update fund scale - fund code: {fundCode}")
        
        sql = f"""
            UPDATE sec_fund
            SET fund_scale={latestFundScale}
            WHERE fund_code='{fundCode}'
            """
        rowcount = cs1.execute(sql)
        conn.commit()
        return 0
    except Exception as e:
        print(e)
        return -1;
    
def loadFund2DB(fundCode, downloadIfNotExists=True):
    #Check data file exists
    srcDataFile = f"{savePath}/{fundCode}-{currentDate}.js"
    print(f"loadFund2DB - srcDataFile: {srcDataFile}")
    try:
        #if not, the download
        checkFile = os.path.isfile(srcDataFile)
        if checkFile==False:
            fLog.writeLog(f"Start download fund info {fundCode}")
            returnCode, srcDataFile = downloadFund(fundCode)
            fLog.writeLog(f"Download fund info {fundCode} {'success.' if returnCode==0 else 'failed!'}")
        else:
        	fLog.writeLog(f"File exists - {srcDataFile}")
        
        #if yes, then load
        content = open(srcDataFile, "r").read()
        jsFund = execjs.compile(content)
        
        #loadNetWorthTrend(jsFund)
        loadFundStockMapping(jsFund)
        updateFundScale(jsFund)
        
        return 0
    except Exception as e:
        fLog.writeLog(f"loadFund2DB exception - {e}")
        return -1
    
def loadFund2DBCallback(returnCode):
    fLog.writeLog(f"Load fund into DB {'success.' if returnCode==0 else 'failed!'}")
    
    
if __name__ == "__main__":
    sqlFundList = f"""
        SELECT fund_code, fund_name 
        FROM sec_fund 
        /* WHERE fund_scale is null */
        WHERE fh_mark is not null
        ORDER BY fund_code
        """
    #loadFund2DB
    sqlFundList = """
        SELECT fund_code, fund_name FROM (
        	SELECT @rownum:=@rownum+1 rownum, a.*
        	FROM sec_fund a, (SELECT @rownum:=0) b
            WHERE fund_type NOT IN ('货币型') AND fund_type NOT LIKE '%债券%'
                AND fund_code NOT IN (
                    SELECT DISTINCT fund_code FROM sec_fund_trend a 
                    WHERE net_date=UNIX_TIMESTAMP('2021-02-09')*1000
                )
        	ORDER BY fund_scale DESC
        ) t WHERE rownum<=9000
        """
    #loadFund2DB
    sqlFundList = """
        SELECT fund_code, fund_name FROM sec_fund a
        WHERE fund_type NOT IN ('货币型') AND fund_type NOT LIKE '%债券%' 
	    AND NOT EXISTS(
            SELECT 1 FROM (
                SELECT DISTINCT fund_code FROM sec_fund_trend
                WHERE net_date>=UNIX_TIMESTAMP(DATE_SUB(CURDATE(), INTERVAL 5 DAY))*1000
            ) b
            WHERE a.fund_code=b.fund_code
	    ) AND fund_code='161725'
        ORDER BY fund_scale DESC
        """
    sqlFundList = """
        SELECT fund_code, fund_name FROM sec_fund a
        WHERE fund_type NOT IN ('货币型') AND fund_type NOT LIKE '%债券%'
        ORDER BY fund_scale DESC
        """
        #AND fund_code='161725'白酒
    #analyzeFundLastNRate
    sqlFundList2 = """
        SELECT f.fund_code, fund_name FROM (
                SELECT DISTINCT fund_code FROM sec_fund_trend a 
                WHERE net_date>=UNIX_TIMESTAMP('2021-02-8')*1000
        ) t JOIN sec_fund f ON t.fund_code=f.fund_code
        WHERE f.last_now_date IS NOT NULL
        """
    rowcount = cs1.execute(sqlFundList)
    conn.commit()
    funds = cs1.fetchall()
    pool = multiprocessing.Pool(processes=5)
    for f in funds:
        fLog.writeLog(f"Start load fund into DB {f[0]} - {f[1]}")
        try:
            #Way 1: single process
            #returnCode = loadFund2DB(f[0])
            #Way 2: multiprocessing
            pool.apply_async(loadFund2DB, (f[0], ), callback=loadFund2DBCallback)
            #pool.apply_async(analyzeFundLastNRate, (f[0], ))
        except Exception as e:
            fLog.writeLog(f"Load fund into DB {f[0]} interrupt with exception {e}.")
    pool.close()
    pool.join()
    fLog.writeLog(f"All processes are completed, multiprocessing pool closed.")

    cs1.close()
    conn.close()
    dbPool.close()
    fLog.writeLog(f"All processes are completed, DB connection and DB pool connections are closed.")
    fLog.close()
