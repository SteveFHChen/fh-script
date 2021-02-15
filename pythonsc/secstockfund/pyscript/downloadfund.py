
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
    
    #Performance tunned, fast get max net date, and fast add not existing into DB
    sqlMaxNetDate = f"""
        SELECT fund_code, MAX(net_date) max_net_date 
        FROM sec_fund_trend 
        WHERE net_date >=UNIX_TIMESTAMP('2021-01-20')*1000 AND fund_code='{fundCode}'
        """
    connx = dbPool.connection()
    csx=connx.cursor()
    rowcount = csx.execute(sqlMaxNetDate)
    result = csx.fetchall()
    maxNetDate = result[0][1]
    
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
    for s in stockCodes:
        sql = f"""
            INSERT INTO sec_fund_stock_mapping(fund_code, stock_code)
            VALUES('{fundCode}', '{s}');
            """
        rowcount = cs1.execute(sql)
    conn.commit()
    return 0

def updateFundScale(jsContent):
    try:
        fundCode = jsContent.eval("fS_code")
        fundScale = jsContent.eval("Data_fluctuationScale")
        latestFundScale = fundScale['series'][-1]['y']
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
        
        #if yes, then load
        content = open(srcDataFile, "r").read()
        jsFund = execjs.compile(content)
        
        loadNetWorthTrend(jsFund)
        #loadFundStockMapping(jsFund)
        #updateFundScale(jsFund)
        
        return 0
    except:
        return -1
    
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
    #analyzeFundLastNRate
    sqlFundList = """
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
            #pool.apply_async(loadFund2DB, (f[0], ))
            pool.apply_async(analyzeFundLastNRate, (f[0], ))
            fLog.writeLog(f"Load fund into DB {f[0]} {'success.' if returnCode==0 else 'failed!'}")
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
