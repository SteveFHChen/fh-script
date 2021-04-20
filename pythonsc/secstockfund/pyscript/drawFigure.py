#Draw figure
from fhsechead import *
from dbpoolutils import *

import multiprocessing

logFile='C:/fh/testenv1/sec/drawFigure.log'
fLog = LogWriter(logFile)

font = {'family':'SimHei', 'weight':'normal', 'size':12} #字体设置方式2，可显示中文字体
legend_loc = 'upper left'

def prepareSingleFundData(fundHistory, index, fund):
    fLog.writeLog(f"Call drawSingleFigure() start for fund {index}-{fund}.")
    connx = dbPool.connection()
    csx=connx.cursor()
    
    sqlMaxNetDate = f"""
        SELECT FROM_UNIXTIME(MAX(net_date)/1000, '%Y-%m-%d') net_date
        FROM sec_fund_trend
        WHERE fund_code = '{fund['fundCode']}'
            AND net_date >=UNIX_TIMESTAMP('2021-01-28')*1000
        """
    rowcount = csx.execute(sqlMaxNetDate)
    connx.commit()
    maxNetDate = csx.fetchall()[0][0]
    
    sqlFundNet = f"""
        select unit_net, day_delta_rate
        FROM sec_fund_trend 
        WHERE fund_code = '{fund['fundCode']}'
            AND net_date >=UNIX_TIMESTAMP('2020-01-01')*1000
        ORDER BY net_date
        """
    if (fund['fundCode'] in fundHistory.keys()) == False: #Improve performance
        rowcount = csx.execute(sqlFundNet)
        connx.commit()
        fundNet = csx.fetchall()
        
        fundNetArray = []
        currRate = 0
        for n in fundNet:
            currRate+=n[1]
            fundNetArray.append(currRate)
        x = np.arange(len(fundNetArray))
        
        fundHistory[fund['fundCode']]={"x":x, "y":fundNetArray, 'maxNetDate': maxNetDate}
        fLog.writeLog(f"B fundHistory keys: {fundHistory.keys()}")
        
    csx.close()
    connx.close()
    fLog.writeLog(f"Call drawSingleFigure() finished for fund {index}-{fund}.")

def drawFundFigure(fundList):
    fundHistory = multiprocessing.Manager().dict() #Python 进程之间共享数据
    pool = multiprocessing.Pool(processes=5)
    plt.figure(figsize=(15, len(fundList)*5), dpi=100)
    for index, fund in enumerate(fundList):
        fLog.writeLog(f"Start to draw figure {index}/{len(fundList)} - {fund}")
        #index, fund = 0, {'fundCode': '003834', 'fundName': '华夏能源革新股票'}
        #Way 1: single process
        #drawSingleFigure(index, fund)
        #Way 2: multiprocessing
        pool.apply_async(prepareSingleFundData, args=(fundHistory, index, fund, ))
    pool.close()
    pool.join()
    fLog.writeLog(f"All processes are completed, multiprocessing pool closed.")
    
    fLog.writeLog(f"fundHistory keys: {fundHistory.keys()}")
    for index, fund in enumerate(fundList):
        x = fundHistory[fund['fundCode']]['x']
        fundNetArray = fundHistory[fund['fundCode']]['y']
        maxNetDate = fundHistory[fund['fundCode']]['maxNetDate']
        
        #Set a window and do data calculation
        last, lastIndex = fundNetArray[-1], x[-1]
        latest, latestX = fundNetArray[-50:], x[-50:]
        maxValue, minValue = max(latest), min(latest)
        maxIndex, minIndex = latest.index(maxValue), latest.index(minValue)
        
        #Long term
        plt.subplot2grid((len(fundList),5), (index,0), colspan=3)
        plt.plot(x, fundNetArray, label=f"{index+1}-{fund['fundCode']}-{fund['fundName']}-{maxNetDate}")
        plt.legend(loc=legend_loc, prop=font);
        
        plt.plot([x[0], x[-1]], [last, last], color='blue', linestyle='--')
        plt.plot([latestX[maxIndex], latestX[maxIndex]], [minValue, maxValue], color='red')
        plt.plot([latestX[minIndex], latestX[minIndex]], [minValue, maxValue], color='green')
        
        #Short term
        plt.subplot2grid((len(fundList),5), (index,3), colspan=2)
        plt.plot(x[-50:], latest, label=f"{fund['fundName']}")
        plt.legend(loc=legend_loc, prop=font);
        
        plt.plot([x[-50], x[-1]], [last, last], color='blue', linestyle='--')
        plt.plot([latestX[maxIndex], latestX[maxIndex]], [minValue, maxValue], color='red')
        plt.plot([latestX[minIndex], latestX[minIndex]], [minValue, maxValue], color='green')
        plt.text(latestX[maxIndex], last, f"{round(last-maxValue,2)}%\n{len(x)-latestX[maxIndex]}Day", color='red', fontsize=20)
        plt.text(latestX[minIndex], last, f"{round(last-minValue,2)}%\n{len(x)-latestX[minIndex]}Day", color='green', fontsize=20)
        
    #plt.show()
    plt.savefig('C:/Users/steve/Desktop/fund4.png',bbox_inches='tight')

def drawFundFigureMain():
    sqlFundList = f"""
        SELECT fund_code, fund_name
        FROM sec_fund 
        WHERE fh_mark IS NOT NULL
        ORDER BY fund_code
        """
    sqlFundList = f"""
        SELECT fund_code, fund_name FROM (
            SELECT @rownum:=@rownum+1 AS rownum, t.* FROM sec_fund t, (SELECT @rownum:=0) k
            WHERE fh_mark IS NOT NULL AND net_date IS NOT NULL
            ORDER BY fund_scale DESC
        ) a WHERE rownum<=10 ORDER BY rownum
        """
    sqlFundList = """
        SELECT fund_code, fund_name FROM (
            SELECT ROUND(lastn_top_rate-lastn_low_rate,3) fudu, 
                   ROUND(lastn_top_rate - last_now_rate,3) godown,
                f.* 
            FROM sec_fund f
        ) t
        WHERE fund_scale>=100 AND fudu>10 AND godown>1
        ORDER BY godown DESC, fudu DESC
        """
    #查询使用高度欢迎股多个的基金
    sqlFundList = """
        SELECT b.fund_code, b.fund_name FROM (
            SELECT m.fund_code, COUNT(1) welcome_stock_cnt
            FROM sec_fund_stock_mapping m
            LEFT JOIN sec_fund f ON m.fund_code=f.fund_code
            LEFT JOIN sec_stock s ON m.stock_code=s.stock_code
            WHERE fund_welcome_cnt>=680
            GROUP BY m.fund_code HAVING COUNT(1)>=8
            ORDER BY 2 DESC, 1
        ) a
        LEFT JOIN sec_fund b ON a.fund_code=b.fund_code
        WHERE fund_scale > 100
        ORDER BY 2 DESC, fund_scale DESC
        """
    #我关注的基金
    sqlFundList = """
        SELECT fund_code, fund_name 
        FROM sec_fund 
        WHERE fh_mark IS NOT NULL AND fh_mark='+' 
        ORDER BY fund_scale DESC
        """
    rowcount = cs1.execute(sqlFundList)
    result = cs1.fetchall()
    fundList = []
    for s in result:
        fundList.append({
            "fundCode": s[0],
            "fundName": s[1]
            })
        
    drawFundFigure(fundList)
    return rowcount
    
    
def prepareSingleStockData(stockHistory, index, stock):
    fLog.writeLog(f"Call drawSingleFigure() start for fund {index}-{stock}.")
    connx = dbPool.connection()
    csx=connx.cursor()
    
    sqlMaxNetDate = f"""
        SELECT price_date
        FROM sec_stock_history
        WHERE stock_code = '{stock['stockCode']}'
            AND price_date >=DATE('2021-01-28')
        """
    rowcount = csx.execute(sqlMaxNetDate)
    connx.commit()
    maxNetDate = csx.fetchall()[0][0]
    
    sqlFundNet = f"""
        select open_price, high_price, low_price, close_price
        FROM sec_stock_history 
        WHERE stock_code = '{stock['stockCode']}'
            AND price_date >=DATE('2020-01-28')
        ORDER BY price_date
        """
    if (stock['stockCode'] in stockHistory.keys()) == False: #Improve performance
        rowcount = csx.execute(sqlFundNet)
        connx.commit()
        fundNet = csx.fetchall()
        
        fundNetArray = []
        currRate = 0
        for n in fundNet:
            currRate=n[1]
            fundNetArray.append(currRate)
        x = np.arange(len(fundNetArray))
        
        stockHistory[stock['stockCode']]={"x":x, "y":fundNetArray, 'maxNetDate': maxNetDate}
        fLog.writeLog(f"B stockHistory keys: {stockHistory.keys()}")
        
    csx.close()
    connx.close()
    fLog.writeLog(f"Call drawSingleFigure() finished for stock {index}-{stock}.")
    
def drawStockFigure(stockList):
    stockHistory = multiprocessing.Manager().dict() #Python 进程之间共享数据
    pool = multiprocessing.Pool(processes=5)
    plt.figure(figsize=(15, len(stockList)*5), dpi=100)
    for index, stock in enumerate(stockList):
        fLog.writeLog(f"Start to draw figure {index}/{len(stockList)} - {stock}")
        #index, stock = 0, {'stockCode': '003834', 'fundName': '华夏能源革新股票'}
        #Way 1: single process
        #drawSingleFigure(index, stock)
        #Way 2: multiprocessing
        pool.apply_async(prepareSingleStockData, args=(stockHistory, index, stock, ))
    pool.close()
    pool.join()
    fLog.writeLog(f"All processes are completed, multiprocessing pool closed.")
    
    fLog.writeLog(f"stockHistory keys: {stockHistory.keys()}")
    for index, stock in enumerate(stockList):
        if stock['stockCode'] not in stockHistory.keys():
            print(f"------------>Stock {stock['stockCode']} has no history data.")
            continue
        x = stockHistory[stock['stockCode']]['x']
        print(f"------------>Stock {stock['stockCode']} - len x: {len(x)}")
        fundNetArray = stockHistory[stock['stockCode']]['y']
        maxNetDate = stockHistory[stock['stockCode']]['maxNetDate']
        
        #Set a window and do data calculation
        last, lastIndex = fundNetArray[-1], x[-1]
        latest, latestX = fundNetArray[-50:], x[-50:]
        maxValue, minValue = max(latest), min(latest)
        maxIndex, minIndex = latest.index(maxValue), latest.index(minValue)
        
        #Long term
        plt.subplot2grid((len(stockList),5), (index,0), colspan=3)
        plt.plot(x, fundNetArray, label=f"{index+1}-{stock['stockCode']}-{stock['stockName']}-{maxNetDate}")
        plt.legend(loc=legend_loc, prop=font);
        
        plt.plot([x[0], x[-1]], [last, last], color='blue', linestyle='--')
        plt.plot([latestX[maxIndex], latestX[maxIndex]], [minValue, maxValue], color='red')
        plt.plot([latestX[minIndex], latestX[minIndex]], [minValue, maxValue], color='green')
        
        #Short term
        plt.subplot2grid((len(stockList),5), (index,3), colspan=2)
        plt.plot(x[-50:], latest, label=f"{stock['stockName']}")
        plt.legend(loc=legend_loc, prop=font);
        
        plt.plot([x[-50], x[-1]], [last, last], color='blue', linestyle='--')
        plt.plot([latestX[maxIndex], latestX[maxIndex]], [minValue, maxValue], color='red')
        plt.plot([latestX[minIndex], latestX[minIndex]], [minValue, maxValue], color='green')
        plt.text(latestX[maxIndex], last, f"{round(last-maxValue,2)}%\n{len(x)-latestX[maxIndex]}Day", color='red', fontsize=20)
        plt.text(latestX[minIndex], last, f"{round(last-minValue,2)}%\n{len(x)-latestX[minIndex]}Day", color='green', fontsize=20)
        
    #plt.show()
    plt.savefig('C:/Users/steve/Desktop/stock4.png',bbox_inches='tight')
    
    
def drawStockFigureMain():
    sqlStockList = """
        SELECT CONCAT(sh_sz_indicator, stock_code) stock_code, stock_name 
        FROM sec_stock 
        WHERE sh_sz_indicator IS NOT NULL AND (stock_name LIKE '%茅台' or stock_name LIKE '%酒')
        ORDER BY 1
        """
    #我关注的基金对应的股票
    sqlStockList = """
        SELECT DISTINCT CONCAT(sh_sz_indicator, s.stock_code) stock_code, s.stock_name
        FROM sec_fund f, sec_fund_stock_mapping m, sec_stock s
        WHERE f.fh_mark IS NOT NULL AND f.fh_mark='+' 
		AND f.fund_code=m.fund_code AND m.stock_code=s.stock_code
		AND s.stock_code IS NOT NULL AND sh_sz_indicator IS NOT NULL
        ORDER BY fund_scale DESC
        """
    #酒股
    sqlStockList = """
        SELECT CONCAT(sh_sz_indicator, s.stock_code) stock_code, s.stock_name
        FROM sec_stock s
        WHERE region_code='CN' AND stock_type='股票' AND industry='食品饮料' 
		AND (fund_welcome_cnt>0 OR stock_name LIKE '%酒%')
        ORDER BY fund_welcome_cnt DESC
        """
    #涨停股
    sqlStockList = """
        SELECT CONCAT(sh_sz_indicator, s.stock_code) stock_code, s.stock_name
        FROM sec_stock s
        WHERE region_code='CN' 
		AND percent >= 0.09
        ORDER BY percent DESC
        """
    rowcount = cs1.execute(sqlStockList)
    result = cs1.fetchall()
    stockList = []
    for s in result:
        stockList.append({
            "stockCode": s[0],
            "stockName": s[1]
            })
        
    drawStockFigure(stockList)
    return rowcount

if __name__ == "__main__":
    startTime = datetime.datetime.now()
    #rowcount=drawFundFigureMain()
    
    #endTime = datetime.datetime.now()
    #fLog.writeLog(f"Draw figure for {rowcount} funds total spent time {(endTime-startTime).seconds}s")
    
    #startTime = datetime.datetime.now()
    rowcount=drawStockFigureMain()
    
    endTime = datetime.datetime.now()
    fLog.writeLog(f"Draw figure for {rowcount} funds/stocks total spent time {(endTime-startTime).seconds}s")
    
    cs1.close()
    conn.close()
    dbPool.close()
    fLog.close()

"""
    Python 进程之间共享数据
    https://www.cnblogs.com/xiaxuexiaoab/p/8558519.html
    最初以为是没添加global声明导致修改未生效，但实际操作发现global方式在多进程中也只能读不能写。
    
    共享方式        支持的类型
    Shared memory	ctypes当中的类型，通过RawValue，RawArray等包装类提供
    Inheritance	    系统内核对象，以及基于这些对象实现的对象。包括Pipe, Queue, JoinableQueue, 同步对象(Semaphore, Lock, RLock, Condition, Event等等)
    Server process	所有对象，可能需要自己手工提供代理对象(Proxy)
    
    进程之间共享数据(数值型)
        num=multiprocessing.Value("d",10.0) # d表示数值,主进程与子进程共享这个value。（主进程与子进程都是用的同一个value） 
    进程之间共享数据(数组型)
        num=multiprocessing.Array("i",[1,2,3,4,5])   #主进程与子进程共享这个数组 
    进程之间共享数据(dict,list)
        mydict=multiprocessing.Manager().dict()   #主进程与子进程共享这个字典  
        mylist=multiprocessing.Manager().list(range(5))   #主进程与子进程共享这个List  
"""