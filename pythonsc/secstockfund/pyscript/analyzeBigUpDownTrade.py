# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 23:21:14 2021

@author: steve
"""

from fhsechead import *

def analyzeBuyStockAtDown10(startDate, endDate, tradeDates):
    print("hello")
    startIndex, endIndex = tradeDates.index(startDate), tradeDates.index(endDate)
    print(startIndex, endIndex)
    for i in range(startIndex, endIndex + 1):
        if i+2+1 > len(tradeDates):
            print(f"Analyze broken on {tradeDates[i]}!")
            break;
        dxm1, dx, dxa1, dxa2 = tradeDates[i-1], tradeDates[i], tradeDates[i+1], tradeDates[i+2]
        print(dxm1, dx, dxa1, dxa2)
        #dx is buy day, dxm1 is dx minus 1 day, dxa1 is dx add 1 day.
        sqlInsertDetail = f"""
           	INSERT INTO sec_stock_tradeatbig(
           		analyze_type, buy_date, stock_code, stock_name, dxm1_percent, dx_percent, buy_price, 
           		sell_price, earn_percent, sell_price2, earn_percent2,
           		gain_indicator, no_indicator, loss_indicator)
           	SELECT '跌停价买入涨停股' analyze_type, '{dx}' buy_date, tx.stock_code, tx.stock_name, 
           		txm1.delta_percent dxm1_percent, tx.delta_percent dx_percent, 
           		tx.low_price buy_price, 
           		txa1.high_price sell_price, ROUND((txa1.high_price-tx.low_price)/tx.low_price*100,0) earn_percent, 
           		txa2.high_price sell_price2, ROUND((txa2.high_price-tx.low_price)/tx.low_price*100,0) earn_percent2, 
           		CASE WHEN ROUND((txa1.high_price-tx.low_price)/tx.low_price*100,0)>0 THEN '赚' ELSE NULL END AS gain_indicator,
           		CASE WHEN ROUND((txa1.high_price-tx.low_price)/tx.low_price*100,0)=0 THEN '平' ELSE NULL END AS no_indicator,
           		CASE WHEN ROUND((txa1.high_price-tx.low_price)/tx.low_price*100,0)<0 THEN '亏' ELSE NULL END AS loss_indicator
           		#, txm1.*
           	FROM (SELECT * FROM sec_stock_history WHERE price_date = '{dxm1}' AND delta_percent>=9) txm1 #dx-1日涨停收盘的股票
           	LEFT JOIN (SELECT * FROM sec_stock_history WHERE price_date = '{dx}' AND low_price<=yest_close*(1-0.09)) tx ON txm1.stock_code=tx.stock_code #dx日以跌停价买入
           	LEFT JOIN (SELECT * FROM sec_stock_history WHERE price_date = '{dxa1}' ) txa1 ON txa1.stock_code=tx.stock_code #dx+1日可卖
           	LEFT JOIN (SELECT * FROM sec_stock_history WHERE price_date = '{dxa2}' ) txa2 ON txa2.stock_code=tx.stock_code #dx+2日可卖
           	WHERE txa1.stock_code IS NOT NULL
            """
        rowcount = cs1.execute(sqlInsertDetail)
        conn.commit()
    print("Analyze analyzeBuyStockAtDown10 completed.")

def buyStockAtUp10(startDate, endDate, tradeDates):
    startIndex, endIndex = tradeDates.index(startDate), tradeDates.index(endDate)
    print(startIndex, endIndex)
    for i in range(startIndex, endIndex + 1):
        if i+2+1 > len(tradeDates):
            print(f"Analyze broken on {tradeDates[i]}!")
            break;
        dxm1, dx, dxa1, dxa2 = tradeDates[i-1], tradeDates[i], tradeDates[i+1], tradeDates[i+2]
        print(dxm1, dx, dxa1, dxa2)
        #dx is buy day, dxm1 is dx minus 1 day, dxa1 is dx add 1 day.
        sqlInsertDetail = f"""
           	INSERT INTO sec_stock_tradeatbig(
           		analyze_type, buy_date, stock_code, stock_name, dxm1_percent, dx_percent, buy_price, 
           		sell_price, earn_percent, sell_price2, earn_percent2,
           		gain_indicator, no_indicator, loss_indicator)
            SELECT '涨停价买入涨停股' analyze_type, '{dx}' buy_date, tx.stock_code, tx.stock_name, 
            	txm1.delta_percent dxm1_percent, tx.delta_percent dx_percent, 
            	tx.open_price buy_price, 
            	txa1.high_price sell_price, ROUND((txa1.high_price-tx.open_price)/tx.open_price*100,0) earn_percent, 
            	txa2.high_price sell_price2, ROUND((txa2.high_price-tx.open_price)/tx.open_price*100,0) earn_percent2, 
            	CASE WHEN ROUND((txa1.high_price-tx.open_price)/tx.open_price*100,0)>0 THEN '赚' ELSE NULL END AS gain_indicator,
            	CASE WHEN ROUND((txa1.high_price-tx.open_price)/tx.open_price*100,0)=0 THEN '平' ELSE NULL END AS no_indicator,
            	CASE WHEN ROUND((txa1.high_price-tx.open_price)/tx.open_price*100,0)<0 THEN '亏' ELSE NULL END AS loss_indicator
            	#, txm1.*
            FROM (SELECT * FROM sec_stock_history WHERE price_date = '{dxm1}' AND delta_percent>=9) txm1 #dx-1日涨停收盘的股票
            LEFT JOIN (SELECT * FROM sec_stock_history WHERE price_date = '{dx}' /*AND low_price<=yest_close*(1-0.09)*/) tx ON txm1.stock_code=tx.stock_code #dx日以跌停价买入
            LEFT JOIN (SELECT * FROM sec_stock_history WHERE price_date = '{dxa1}' ) txa1 ON txa1.stock_code=tx.stock_code #dx+1日可卖
            LEFT JOIN (SELECT * FROM sec_stock_history WHERE price_date = '{dxa2}' ) txa2 ON txa2.stock_code=tx.stock_code #dx+2日可卖
            WHERE txa1.stock_code IS NOT NULL
            """
        rowcount = cs1.execute(sqlInsertDetail)
        conn.commit()
    print("Analyze analyzeBuyStockAtDown10 completed.")

def getTradeDate(startDate, endDate):
    """
        Parameters sample:
        startDate, endDate = '2021-01-01', '2021-04-16'
    """
    sqlGetTradeDate = f"""
        SELECT DATE_FORMAT(price_date, '%Y-%m-%d') trade_date
        FROM sec_stock_history 
        WHERE stock_code='0600519' 
        	AND price_date BETWEEN 
                DATE_SUB('{startDate}', INTERVAL 10 DAY) AND  DATE_ADD('{endDate}', INTERVAL 10 DAY)
        ORDER BY price_date
        """
    rowcount = cs1.execute(sqlGetTradeDate)
    records = cs1.fetchall()
    tradeDates = [r[0] for r in records] #transfer tuple to list
    return tradeDates

if __name__ == "__main__":
    startDate, endDate ='2021-01-04', '2021-04-16'
    
    tradeDates = getTradeDate(startDate, endDate)
    #analyzeBuyStockAtDown10('2021-04-13', '2021-04-13', tradeDates)
    analyzeBuyStockAtDown10(startDate, endDate, tradeDates)
    buyStockAtUp10(startDate, endDate, tradeDates)
    
    cs1.close()
    conn.close()