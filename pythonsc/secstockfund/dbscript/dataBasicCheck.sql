SELECT COUNT(1) FROM sec_fund_company;#161
SELECT COUNT(1) FROM sec_fund;#11910
SELECT COUNT(1) FROM sec_stock;#21006
SELECT COUNT(1) FROM sec_fund_stock_mapping;#63356
SELECT COUNT(1) FROM sec_fund_trend; #8669009

#基金分类
SELECT fund_type, COUNT(1) FROM sec_fund GROUP BY fund_type ORDER BY fund_type;
/* 
ETF-场内    366       
QDII          213       
QDII-ETF      21        
QDII-指数   89        
债券型     2462      
债券指数  256       
其他创新  3         
分级杠杆  28        
固定收益  37        
定开债券  1091      
混合-FOF    178       
混合型     4850      
理财型     13        
联接基金  388       
股票-FOF    1         
股票型     597       
股票指数  620       
货币型     697       
 */

SELECT region_code, COUNT(1) FROM sec_stock GROUP BY region_code ORDER BY 1;
/* 
CN           3935      
HK           2010      
US           15061     
 */
SELECT region_code, section_name, COUNT(1) 
FROM sec_stock 
GROUP BY region_code, section_name
ORDER BY 1,2,3;
/* 
CN                         264       
CN           A股          3563      
CN           B股          108       
HK           港股        2010      
US                         15061     
 */

SELECT region_code, section_name, subsection_name, COUNT(1) 
FROM sec_stock 
GROUP BY region_code, section_name, subsection_name
ORDER BY 1,2,3;
/* 
CN                                          264       
CN           A股          创业板        729       
CN           A股          指数           1         
CN           A股          沪A             1461      
CN           A股          深A             1372      
CN           B股          沪B             54        
CN           B股          深B             54        
HK           港股        股票           2010      
US                                          15061     
 */

SELECT section_name, COUNT(1) 
FROM sec_stock_section_pivot 
GROUP BY section_name
ORDER BY 1;
/* 
industry      4275      
region        4274      
 */

SELECT section_name, stock_code, COUNT(1)
FROM sec_stock_section_pivot
#where section_name='industry'
GROUP BY section_name, stock_code HAVING COUNT(1)>=2;
#每个股票在一个分类标准中只会存在于一个分类，不会跨2个分类

SELECT section_name, section_value, COUNT(1) 
FROM sec_stock_section_pivot 
#WHERE section_name='region'
GROUP BY section_name, section_value 
ORDER BY 1,3 DESC;

UPDATE sec_stock a
SET industry = (
	SELECT section_value
	FROM sec_stock_section_pivot b
	WHERE section_name='industry'
	  AND a.stock_code=b.stock_code
)
WHERE EXISTS (
	SELECT section_value
	FROM sec_stock_section_pivot b
	WHERE section_name='industry'
	  AND a.stock_code=b.stock_code
);

UPDATE sec_stock a
SET province= (
	SELECT section_value
	FROM sec_stock_section_pivot b
	WHERE section_name='region'
	  AND a.stock_code=b.stock_code
)
WHERE EXISTS (
	SELECT section_value
	FROM sec_stock_section_pivot b
	WHERE section_name='region'
	  AND a.stock_code=b.stock_code
);