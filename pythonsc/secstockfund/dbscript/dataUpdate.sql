
#@#downloadStock.py
/*
conda activate pytorch
cd C:/fh/ws/ws1/fh-script/pythonsc/secstockfund/pyscript
 */
#python C:/fh/ws/ws1/fh-script/pythonsc/secstockfund/pyscript/downloadStock.py
#Capture all CN stocks
SELECT COUNT(1) FROM sec_stock WHERE region_code='CN';

#@#downloadSecSectionCategory.py
#python C:/fh/ws/ws1/fh-script/pythonsc/secstockfund/pyscript/downloadSecSectionCategory.py
SELECT * FROM sec_stock WHERE region_code='CN' AND section_name IS NULL;#120 records
SELECT DISTINCT region_code FROM sec_stock;
SELECT * FROM sec_stock WHERE stock_code IN ('000001') OR stock_name ='五粮液';
SELECT DISTINCT section_name FROM  sec_stock_section_pivot;

#@#补充或更新股票所属行业
UPDATE sec_stock s
SET industry = (
	SELECT section_value
	FROM sec_stock_section_pivot p
	WHERE p.section_name='industry' AND s.stock_code=p.stock_code
)
WHERE s.region_code='CN' AND s.stock_type='股票' 
	AND EXISTS(
		SELECT 1
		FROM sec_stock_section_pivot p
		WHERE p.section_name='industry' AND s.stock_code=p.stock_code
	);
#@#检查还有哪些股没有行业信息
SELECT * FROM sec_stock WHERE region_code='CN' AND stock_type<>'指数' AND industry IS NULL;

#@#补充或更新股票所属省份
UPDATE sec_stock s
SET province = (
	SELECT section_value
	FROM sec_stock_section_pivot p
	WHERE p.section_name='region' AND s.stock_code=p.stock_code
)
WHERE s.region_code='CN' AND s.stock_type='股票' 
	AND EXISTS(
		SELECT 1
		FROM sec_stock_section_pivot p
		WHERE p.section_name='region' AND s.stock_code=p.stock_code
	);
#@#检查还有哪些股没有省份信息
SELECT * FROM sec_stock WHERE region_code='CN' AND stock_type<>'指数' AND province IS NULL; #and industry IS NOT NULL;

#@#查看股票表中stock_code重叠的（查询发现只有A股的指数和B股的股票有重叠）
SELECT * FROM sec_stock a WHERE EXISTS (
	SELECT 1 FROM (
		SELECT stock_code, COUNT(1)
		FROM sec_stock 
		GROUP BY stock_code HAVING COUNT(1)>=2
	) b WHERE a.stock_code=b.stock_code
)
ORDER BY 1,2
#order by stock_type, 1
;

SELECT section_name, stock_type, COUNT(1)
FROM sec_stock
WHERE region_code='CN'
GROUP BY section_name, stock_type;

#@#downloadStockHistory.py
#python C:/fh/ws/ws1/fh-script/pythonsc/secstockfund/pyscript/downloadStockHistory.py
SELECT COUNT(1) FROM sec_stock_history WHERE price_date>=DATE_SUB(CURDATE(), INTERVAL 1 DAY);
SELECT * FROM sec_stock_history WHERE price_date>=DATE_SUB(CURDATE(), INTERVAL 10 DAY) AND stock_code='0600519';

        SELECT #count(1)
		CONCAT(sh_sz_indicator, stock_code) stock_code, stock_name 
        FROM sec_stock a
        WHERE sh_sz_indicator IS NOT NULL #AND stock_name LIKE '%茅台%'
            AND NOT EXISTS(
                SELECT 1 FROM (
			SELECT DISTINCT stock_code FROM sec_stock_history 
			WHERE price_date>=DATE_SUB(CURDATE(), INTERVAL 2 DAY)
		) b
                WHERE CONCAT(a.sh_sz_indicator, a.stock_code)=b.stock_code
            )
        ORDER BY 1 DESC;
        
        SELECT CONCAT(sh_sz_indicator, stock_code) stock_code, stock_name 
        FROM sec_stock a
        WHERE sh_sz_indicator IS NOT NULL 
            AND NOT EXISTS(
                SELECT 1 FROM (
			SELECT DISTINCT stock_code FROM sec_stock_history 
			WHERE price_date>=DATE_SUB(CURDATE(), INTERVAL 2 DAY)
		) b
                WHERE CONCAT(a.sh_sz_indicator, a.stock_code)=b.stock_code
            )
        ORDER BY 1;
        
SHOW PROCESSLIST;

#@#downloadFundStockCurrentValue.py
#python C:/fh/ws/ws1/fh-script/pythonsc/secstockfund/pyscript/downloadFundStockCurrentValue.py
SELECT * FROM sec_stock ORDER BY price_time DESC;
SELECT COUNT(1) FROM sec_stock WHERE price_time>=DATE_SUB(CURDATE(), INTERVAL 0 DAY);
SELECT * FROM sec_stock WHERE price_time<DATE_SUB(CURDATE(), INTERVAL 0 DAY);

#@#检查股票当前数据更新程度
#1）已经更新的
SELECT COUNT(1)
FROM sec_stock
WHERE region_code='CN' AND price_time>=DATE_SUB(CURDATE(), INTERVAL 1 DAY);
#2）没有更新到的
SELECT COUNT(1)
FROM sec_stock
WHERE region_code='CN' AND (percent IS NULL OR price_time<DATE_SUB(CURDATE(), INTERVAL 1 DAY));

#@#检查还有多少股票的历史价格没有更新到最新数据
SELECT  #count(1) 
	#CONCAT(sh_sz_indicator, stock_code) stock_code, stock_name 
	*
FROM sec_stock a
WHERE sh_sz_indicator IS NOT NULL 
    AND NOT EXISTS(
	SELECT 1 FROM (
		SELECT DISTINCT stock_code FROM sec_stock_history 
		WHERE price_date>=DATE_SUB(CURDATE(), INTERVAL 1 DAY)
	) b
	WHERE CONCAT(a.sh_sz_indicator, a.stock_code)=b.stock_code
    )
ORDER BY 1;

SELECT COUNT(1)
FROM sec_stock_history 
WHERE price_date>=DATE_SUB(CURDATE(), INTERVAL 1 DAY);

#@#downloadFundStockCurrentValue.py
#python C:/fh/ws/ws1/fh-script/pythonsc/secstockfund/pyscript/downloadFundStockCurrentValue.py
SELECT * FROM sec_fund ORDER BY net_date DESC;
SELECT COUNT(1) FROM sec_fund WHERE net_date>=DATE_SUB(CURDATE(), INTERVAL 2 DAY);

#@#检查还有多少基金的历史价格没有更新到最新数据
SELECT COUNT(1) #fund_code, fund_name 
FROM sec_fund a
WHERE fund_type NOT IN ('货币型') AND fund_type NOT LIKE '%债券%' 
    AND NOT EXISTS(
	    SELECT 1 FROM (
		SELECT DISTINCT fund_code FROM sec_fund_trend
		WHERE net_date>=UNIX_TIMESTAMP(DATE_SUB(CURDATE(), INTERVAL 5 DAY))*1000
	    ) b
	    WHERE a.fund_code=b.fund_code
    );
ORDER BY fund_scale DESC;

SELECT COUNT(1)
FROM sec_fund_trend 
WHERE net_date>=UNIX_TIMESTAMP(DATE_SUB(CURDATE(), INTERVAL 2 DAY))*1000;

#
CREATE TABLE sec_fund_stock_mapping_20210402 AS SELECT * FROM sec_fund_stock_mapping;
TRUNCATE TABLE sec_fund_stock_mapping;
#python C:/fh/ws/ws1/fh-script/pythonsc/secstockfund/pyscript/downloadfund.py
SELECT COUNT(1) FROM sec_fund_trend WHERE net_date >=UNIX_TIMESTAMP('2021-04-02')*1000;
SELECT COUNT(1) FROM sec_fund_stock_mapping;

#@#downloadFundStockPosition.py
#python C:/fh/ws/ws1/fh-script/pythonsc/secstockfund/pyscript/downloadFundStockPosition.py
SELECT * FROM sec_fund_stock_mapping 
WHERE fund_percent IS NOT NULL
;

SELECT * FROM sec_fund WHERE fund_code='161725';
SELECT * FROM sec_fund_stock_mapping WHERE fund_code IN ('161725', '000834','000044');
SELECT * FROM sec_fund_stock_mapping WHERE fund_code='161725';
SELECT * FROM sec_fund_stock_mapping_20210402 WHERE fund_code='161725';

#@#查看总共有多少个基金与股票有映射
SELECT COUNT(1) FROM (
	#检查每个基金映射的重仓股票是否是10个
	SELECT fund_code, COUNT(1)
	FROM sec_fund_stock_mapping#_20210402
	GROUP BY fund_code
	ORDER BY 2 DESC
) t;

SELECT *
FROM sec_stock 
WHERE region_code='CN'
ORDER BY fund_welcome_cnt DESC;

SELECT industry, region_code, COUNT(1) 
FROM sec_stock 
#where region_code='CN' 
GROUP BY industry, region_code
ORDER BY 3 DESC;



TRUNCATE TABLE sec_stock_section_pivot;
SELECT section_name, COUNT(1) FROM sec_stock_section_pivot GROUP BY section_name ORDER BY 1;

#@#在概念板块中，发现一个股会打有多个概念标签，
SELECT * 
FROM sec_stock_section_pivot 
WHERE section_name='concept' 
AND stock_code='600519' 
#and section_value='白酒'
ORDER BY stock_code, section_value;
#如何恰当的找出一天中概念板块的热点和冷点可成为一个研究课题。

SELECT section_value, COUNT(1)
FROM sec_stock_section_pivot
WHERE section_name='concept' AND section_value NOT IN (
	SELECT section_value
	FROM sec_stock_section_pivot 
	WHERE section_name='concept' 
	AND stock_code='600519' AND section_value<>'白酒'
)
GROUP BY section_value
ORDER BY 2 DESC, 1;

SELECT * FROM sec_stock_section_pivot p
LEFT JOIN sec_stock s ON p.stock_code=s.stock_code
WHERE section_value='机构重仓' AND s.stock_type='股票' AND s.region_code='CN'
ORDER BY fund_welcome_cnt DESC;

UPDATE sec_stock SET stock_name='中国中免' WHERE stock_code='601888' AND stock_type='股票';


select * from sec_stock_history where stock_code='0600844' order by price_date desc;
#http://quotes.money.163.com/service/chddata.html?code=0600844

