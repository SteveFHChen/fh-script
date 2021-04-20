#C:\fh\ws\ws1\fh-script\pythonsc\secstockfund\pyscript
from fhsechead import *

def analyzeStockContinuity(stock):
    print(f"Start to analyze continuity [{stock['stockName']}]...")
    stockCode = stock['stockCode']
    sql = f"""
        SELECT price_date, stock_code, stock_name, close_price, high_price, low_price, open_price, delta_percent
        FROM sec_stock_history 
        WHERE stock_code='0600526' AND price_date>=DATE_SUB(CURDATE(), INTERVAL 2 YEAR) 
        ORDER BY price_date DESC
        """
        
    sql = f"""
        SELECT DATE_FORMAT(price_date, '%Y%m%d') , IFNULL(ROUND(delta_percent,0), 0) delta_percent,
            open_price, close_price
        FROM sec_stock_history 
        WHERE stock_code='{stockCode}' AND price_date>=DATE_SUB(CURDATE(), INTERVAL 60 DAY) 
        ORDER BY price_date DESC
        """
        
    rowcount = cs1.execute(sql)
    history = cs1.fetchall()

    i = 0
    isAnalyzing = False
    afterUp9PercentIndex, endIndex, startUp9Index, startIndex = None, None, None, None
    maxSkipLimit = 0
    skipLimit, isSkipSuccess = maxSkipLimit, False
    sumPercent, up9Cnt = 0, 0
    result = []
    while i < len(history):
        if history[i][1]>=9: #大涨
            if isAnalyzing==False:
                isAnalyzing, endIndex, afterUp9PercentIndex = True, i, i-1
                print(f'{history[i][0]} 大涨{history[i][1]}%, 开始大涨分析...')
            else:
                print(f'{history[i][0]} 大涨{history[i][1]}%，连续大涨{i-endIndex+1}天')
            up9Cnt += 1
            startUp9Index = i
            sumPercent += history[i][1]
        elif isAnalyzing==True: #正处于连续性分析阶段
            if history[i][1]<9 and history[i][1]>=0: #0%~9%
                print(f'{history[i][0]} 小涨{history[i][1]}%，连续涨{i-endIndex+1}天')
                sumPercent += history[i][1]
            else: #<0%
                startIndex=i-1 #记录暂时性开始点
                while skipLimit > 0: #可连续跳过进行容错或忽略偶尔性的跌
                    i+=1 #继续向前查找
                    if history[i][1]>=0: #查找成功
                        isSkipSuccess = True
                        break
                    else:
                        skipLimit-=1
                skipLimit = maxSkipLimit #Reset
                if isSkipSuccess==True:
                    print(f'{history[i][0]} 跳越跌后发现涨{history[i][1]}%，连续涨{i-endIndex+1}天')
                    sumPercent += sum([history[j][1] for j in range(startIndex+1, i+1)]) #累加被跳越天里的涨跌百分比
                    isSkipSuccess = False
                else:#连续性分析段结束
                    print(f'{history[startIndex+1][0]} 下跌{history[startIndex+1][1]}%，大涨前最后一跌')
                    print(f'连续性大涨分析段：{startIndex - endIndex + 1}天大涨累计{sumPercent}%, {up9Cnt}个涨停，{history[startIndex][0]}至{history[endIndex][0]}.')
                    result += [{
                        'stockCode': stockCode,
                        'stockName': stock['stockName'],
                        
                        'tradeDays': startIndex - endIndex + 1,
                        'beforeUp9Days': startIndex - startUp9Index, 
                        'up9Days': up9Cnt,
                        'afterUp9Percent': history[afterUp9PercentIndex][1] if i>=0 else -999, 
                        'accPercent': sumPercent,
                        'avgPercent': round(sumPercent/(startIndex - endIndex + 1), 1),
                        'realPercent': round((history[endIndex][2] - history[startIndex][2]) / history[startIndex][2] * 100, 0) if history[startIndex][2]!=0 else -999,
                        
                        'closePrice': history[endIndex][2],
                        'highPrice': 0,
                        'lowPrice': 0,
                        'openPrice': history[startIndex][2],
                        
                        'startDate': history[startIndex][0],
                        'endDate': history[endIndex][0],
                        'startUp9Date': history[startUp9Index][0]
                        }]
                    #print(result) #Debug
                    isAnalyzing, sumPercent, up9Cnt = False, 0, 0
        i+=1

    #将分析结果一次性插入数据库中保存
    for r in result:
        sqlSaveResult = f"""
            INSERT INTO sec_stock_continuity(
            	analyze_type, stock_code, stock_name, 
            	trade_days, bf_up9_days, up9_days, af_up9_percent, sum_percent, avg_percent, real_percent, 
            	close_price, high_price, low_price, open_price, 
            	start_date, end_date, st_up9_date)
            VALUES(
            	'大涨', '{r['stockCode']}', '{r['stockName']}',
            	{r['tradeDays']}, {r['beforeUp9Days']}, {r['up9Days']}, {r['afterUp9Percent']}, {r['accPercent']}, {r['avgPercent']}, {r['realPercent']},
            	{r['closePrice']}, {r['highPrice']}, {r['lowPrice']}, {r['openPrice']},
            	'{r['startDate']}', '{r['endDate']}', {r['startUp9Date']}
            )
            """
        rowcount = cs1.execute(sqlSaveResult)
    conn.commit()
    
    
    
"""
elif history[i][1] <= -8:
    print(f'{history[i][0]} 大跌{history[i][1]}%')
"""

if __name__=="__main__":
    sqlStockList = f"""
        SELECT CONCAT(sh_sz_indicator, stock_code) stock_code, stock_name
        FROM sec_stock 
        WHERE region_code='CN' AND stock_type='股票' AND price_time>='2021-04-09'
        ORDER BY stock_code
        """
        #and stock_code in ('600526', '000011', '600519')
    sqlStockList = """
        SELECT stock_code, stock_name FROM sec_stock_history WHERE price_date='2021-04-09' AND delta_percent>=9
        UNION
        SELECT CONCAT(sh_sz_indicator, stock_code) stock_code, stock_name FROM sec_stock WHERE price_time>='2021-04-12' AND percent >=0.09
        ORDER BY stock_code
        """
    rowcount = cs1.execute(sqlStockList)
    records = cs1.fetchall()
    stockList = []
    for r in records:
        stockList += [{'stockCode':r[0], 'stockName':r[1]}]
    
    #Test case
    stockListx=[
        {'stockCode':'0600526', 'stockName':'菲达环保'},
        {'stockCode':'1000011', 'stockName':'深物业A'},
        {'stockCode':'0600519', 'stockName':'贵州茅台'},
        #9 Apr up 10%
        {'stockCode':'0600569', 'stockName':'安阳钢铁'},
        {'stockCode':'1000966', 'stockName':'长源电力'},
        {'stockCode':'1002053', 'stockName':'云南能投'},
        {'stockCode':'1002639', 'stockName':'雪人股份'},
        {'stockCode':'1002762', 'stockName':'金发拉比'},
        ]
    
    for si, stock in enumerate(stockList):
        print(f'[{si}/{len(stockList)}] {stock}')
        analyzeStockContinuity(stock)
    
    cs1.close()
    conn.close()

#技术改进
"""
tuple转dict再使用数据可提高程序的可读性和后期可修改性。
使用多进程同时分析多个股票。
将近2年的股票历史数据抽到一张表中专供历史数据分析，提高历史数据获取效率。

如何以最快的速度将当天的股票价格交易数据以增量方式补充到股票历史数据表中？
如何以增量方式将当天的数据补充进去？

发现股票有大涨，但没有在系统中存在该股，要引起重视，查明原因，将这一类分问一次性解决。
1）新股从以下网站获取
http://quote.eastmoney.com/center/gridlist.html#newshares
为提高性能，可根据日期进行增量补充

制定规则检验方法，并检验规则的准确性。
"""