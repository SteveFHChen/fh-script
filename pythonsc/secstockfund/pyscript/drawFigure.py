#Draw figure
from fhsechead import *

logFile='C:/fh/testenv1/sec/drawFigure.log'
fLog = LogWriter(logFile)

font = {'family':'SimHei', 'weight':'normal', 'size':12} #字体设置方式2，可显示中文字体
legend_loc = 'upper left'

fundHistory = {}
def drawFigure(fundList):
    plt.figure(figsize=(15, len(fundList)*5), dpi=100)
    for index, fund in enumerate(fundList):
        fLog.writeLog(f"Start to draw figure {index}/{len(fundList)} - {fund}")
        #index, fund = 0, {'fundCode': '003834', 'fundName': '华夏能源革新股票'}
        sqlMaxNetDate = f"""
            SELECT FROM_UNIXTIME(MAX(net_date)/1000, '%Y-%m-%d') net_date
            FROM sec_fund_trend
            WHERE fund_code = '{fund['fundCode']}'
                AND net_date >=UNIX_TIMESTAMP('2021-01-28')*1000
            """
        rowcount = cs1.execute(sqlMaxNetDate)
        conn.commit()
        maxNetDate = cs1.fetchall()[0][0]
        
        sqlFundNet = f"""
            select unit_net, day_delta_rate
            FROM sec_fund_trend 
            WHERE fund_code = '{fund['fundCode']}'
            	AND net_date >=UNIX_TIMESTAMP('2020-01-01')*1000
            ORDER BY net_date
            """
        if (fund['fundCode'] in fundHistory.keys()) == False: #Improve performance
            rowcount = cs1.execute(sqlFundNet)
            conn.commit()
            fundNet = cs1.fetchall()
            
            fundNetArray = []
            currRate = 0
            for n in fundNet:
                currRate+=n[1]
                fundNetArray.append(currRate)
            x = np.arange(len(fundNetArray))
            
            fundHistory[fund['fundCode']]={"x":x, "y":fundNetArray}
            
        x = fundHistory[fund['fundCode']]['x']
        fundNetArray = fundHistory[fund['fundCode']]['y']
        
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

if __name__ == "__main__":
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
    rowcount = cs1.execute(sqlFundList)
    result = cs1.fetchall()
    fundList = []
    for s in result:
        fundList.append({
            "fundCode": s[0],
            "fundName": s[1]
            })
        
    drawFigure(fundList)

    cs1.close()
    conn.close()
    fLog.close()
