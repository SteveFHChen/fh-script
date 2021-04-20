#@#查看龙头股 - 哪些股受基金的欢迎度高，哪些股不受基金欢迎
/*方式一：先一次计算，再多次查询，结果有保存，性能高*/

#backup the previous welcome
ALTER TABLE sec_stock ADD fund_wel_cnt_2020 INT;
ALTER TABLE sec_stock MODIFY fund_wel_cnt_2020 INT AFTER fund_welcome_cnt;
UPDATE sec_stock SET fund_wel_cnt_2020 = fund_welcome_cnt WHERE fund_welcome_cnt IS NOT NULL;
#clear to set new welcome
UPDATE sec_stock SET fund_welcome_cnt = NULL WHERE fund_welcome_cnt IS NOT NULL;


UPDATE sec_stock
SET fund_map_code = 
	CASE 
		WHEN stock_code IN ('002685', '002855', '002885', '002915', 'AZN', 'RIO') THEN stock_code#特殊个例特殊处理
		WHEN subsection_name LIKE '沪%' OR (region_code='CN' AND section_name IS NULL) THEN CONCAT(stock_code, '1')#沪股后补1
		WHEN subsection_name LIKE '深%' OR subsection_name='创业板' THEN CONCAT(stock_code, '2')#深股后补2
		WHEN region_code='US' THEN CONCAT(stock_code, '7')#美股后补7
		ELSE stock_code
	END;

UPDATE sec_stock a
SET fund_welcome_cnt=(
	SELECT fund_welcome_cnt FROM (
		SELECT region_stock_code, COUNT(1) fund_welcome_cnt, ROUND(SUM(fund_percent),2) fund_welcome_percent
		FROM sec_fund_stock_mapping
		GROUP BY region_stock_code
	) t1 
	WHERE t1.region_stock_code = a.fund_map_code 
), fund_welcome_percent=(
	SELECT fund_welcome_percent FROM (
		SELECT region_stock_code, COUNT(1) fund_welcome_cnt, ROUND(SUM(fund_percent),2) fund_welcome_percent
		FROM sec_fund_stock_mapping
		GROUP BY region_stock_code
	) t1 
	WHERE t1.region_stock_code = a.fund_map_code
)
WHERE EXISTS(
	SELECT region_stock_code, cnt FROM (
		SELECT region_stock_code, COUNT(1) cnt
		FROM sec_fund_stock_mapping
		GROUP BY region_stock_code
	) t2 
	WHERE t2.region_stock_code = a.fund_map_code
);

#@#查看被基金重仓的股票
SELECT * FROM sec_stock
#where region_code<>'CN'#查看被重仓了非CN的股票
ORDER BY fund_welcome_cnt DESC;

SELECT t1.*, t2.map_stock_cnt
FROM (
	#查看基金有哪些类型，分布比例如何
	SELECT fund_type, COUNT(1) total_cnt FROM sec_fund GROUP BY fund_type
) t1
LEFT JOIN (
	#查看哪些类型的基金有投资股票，分布比例如何
	SELECT fund_type, COUNT(1) map_stock_cnt
	FROM sec_fund f, (SELECT DISTINCT fund_code FROM sec_fund_stock_mapping) m 
	WHERE f.fund_code=m.fund_code GROUP BY fund_type
) t2 ON t1.fund_type=t2.fund_type
ORDER BY 2 DESC,3,1;


#@#查看重仓非CN股票的基金
SELECT s.region_code, s.stock_name, f.* 
FROM sec_fund f, sec_fund_stock_mapping m, sec_stock s
WHERE f.fund_code=m.fund_code AND m.stock_code=s.fund_map_code 
	AND s.region_code<>'CN'
ORDER BY s.stock_name, f.fund_name;

#@#查看有重仓非CN股票的基金所重仓的前10股票
SELECT f.fund_name, s.stock_name, s.region_code
FROM sec_fund f, sec_fund_stock_mapping m, sec_stock s
WHERE f.fund_code=m.fund_code AND m.stock_code=s.fund_map_code 
	AND EXISTS (
		SELECT 1 FROM (
		SELECT f.fund_code
		FROM sec_fund f, sec_fund_stock_mapping m, sec_stock s
		WHERE f.fund_code=m.fund_code AND m.stock_code=s.fund_map_code 
			AND s.region_code<>'CN'
		) t2 WHERE m.fund_code=t2.fund_code
	)
ORDER BY f.fund_name, s.stock_name;

#@#导出基金重仓股票数据
SELECT region_code, section_name, subsection_name, stock_code, stock_name, 
	(IFNULL(fund_welcome_cnt,0)-IFNULL(fund_wel_cnt_2020,0)) delta, fund_welcome_cnt, fund_wel_cnt_2020, 
	industry, now_price, open_price, ROUND(percent*100,0) percent, province, price_time
FROM sec_stock 
WHERE (fund_welcome_cnt IS NOT NULL OR fund_wel_cnt_2020 IS NOT NULL) 
	#and stock_name='紫金矿业'
	#and industry = '有色金属'
ORDER BY fund_welcome_cnt DESC, fund_wel_cnt_2020 DESC, stock_name;#查看基金对所有股票的重仓数
#order by delta desc;
ORDER BY industry, fund_welcome_cnt DESC, fund_wel_cnt_2020 DESC;#查看各行业基金对股票的重仓数，据此快速了解各行业的龙头股、白马股、潜力股

#@#统计基金重仓股排名中各行业前10的股
SELECT  
	region_code, section_name, subsection_name, stock_code, stock_name, rn,
	(IFNULL(fund_welcome_cnt,0)-IFNULL(fund_wel_cnt_2020,0)) delta, fund_welcome_cnt, fund_wel_cnt_2020, 
	industry, now_price, open_price, ROUND(percent*100,0) percent, province, price_time
FROM (
	SELECT IF(@industry=industry, @rowno:=@rowno + 1, @rowno:=1) AS rn, @industry:=industry, s.*
	FROM sec_stock s, (SELECT @rowno:=0, @industry:=NULL) r
	WHERE (fund_welcome_cnt IS NOT NULL OR fund_wel_cnt_2020 IS NOT NULL)
	ORDER BY industry, fund_welcome_cnt DESC, fund_wel_cnt_2020 DESC
) t 
WHERE rn<=10
	#and percent>=0.06#各行业中前10大涨或大跌的股
;

#@#查看概念板块的涨跌情况，并了解明细股，以及是否是基金欢迎股。
#一个股可属于多个概念板块，避免了仅按行业分类至使分类不合理找不到共同点的现象，建议使用聚类算法。
SELECT a.cnt, b.*
FROM (
	#查看概念板块的涨跌情况
	SELECT #* 
		section_value, COUNT(1) cnt
	FROM sec_stock s, sec_stock_section_pivot p 
	WHERE p.section_name='concept' AND s.stock_code=p.stock_code AND price_time>CURDATE() AND percent>0.09 
	GROUP BY section_value
	ORDER BY cnt DESC
) a 
LEFT JOIN (
	#查看概念板块的涨跌名细
	SELECT p.section_value, s.*
	FROM sec_stock_section_pivot p, sec_stock s
	WHERE p.section_name='concept' AND s.stock_code=p.stock_code AND price_time>CURDATE() AND percent>0.09 
	ORDER BY p.section_value, fund_welcome_cnt DESC
) b ON a.section_value=b.section_value
ORDER BY a.cnt DESC, b.section_value, b.fund_welcome_cnt DESC;

#@#查看它们的整体涨跌幅度，了解大涨大跌主要集中在哪个行业或哪个概念板块
SELECT * 
FROM (
	#按每个概念板块查看龙头股——基金欢迎度前10
	SELECT *
	FROM (
		SELECT 
			IF(@concept=section_value, @rowno:=@rowno+1, @rowno:=1) AS rn, @concept:=section_value,
			t.*
		FROM (
		SELECT p.section_value, s.*
		FROM sec_stock_section_pivot p, sec_stock s
		WHERE p.section_name='concept' AND s.stock_code=p.stock_code
		ORDER BY section_value, fund_welcome_cnt DESC
		) t, (SELECT @rowno:=0, @concept:=NULL) r
	) t2 WHERE rn<=10 #and section_value='白酒'
) t3 WHERE percent >=0.06
#order by industry, section_value
ORDER BY section_value, industry
;

#@#查看大涨大跌行业或概念板块的所有股涨跌信息
SELECT p.section_value, s.*
FROM sec_stock_section_pivot p, sec_stock s
WHERE p.section_name='concept' AND s.stock_code=p.stock_code
	AND (section_value ='氢能源' OR industry='交通运输1')
ORDER BY ROUND(percent,2) DESC, section_value, stock_name;

SELECT * FROM sec_stock_section_pivot p WHERE stock_code='002639';
SELECT * FROM sec_fund_stock_mapping WHERE stock_code='002639';
SELECT * FROM sec_stock WHERE province LIKE '%陕%' AND (industry LIKE '%休闲服务%' OR stock_name LIKE '%曲江%');

SELECT region_code, industry, COUNT(1) cnt, SUM(fund_welcome_cnt), SUM(fund_wel_cnt_2020)
FROM sec_stock
WHERE fund_welcome_cnt>=50
#where fund_wel_cnt_2020>=500
GROUP BY region_code, industry
ORDER BY cnt DESC, industry;


SELECT * FROM sec_stock_history WHERE stock_code='0600519' ORDER BY price_date DESC;

#@#股票连续性大涨分析，发现股票中当前正在大幅上升的股。
#对于开始连续涨停2-3天的股票(第一天涨幅超过5%说明是开始要起飞)，很大机会是要连接涨停5天，新股可能更长，发现这样的股票可以轻松一天赚10%。
#对于刚涨停一天的股票，一般第2天都会大跌回原位或跌停。
#连接性分析中，如果几天连续涨停中有一个是没有涨停甚至是大跌了，应当忽略这天的数据。
SELECT t2.delta_percent, #t3.delta_percent, 
	t1.*
FROM (SELECT * FROM sec_stock_history WHERE price_date='2021-03-15' AND delta_percent>=9) t1
JOIN (SELECT * FROM sec_stock_history WHERE price_date='2021-03-16' AND delta_percent>=9) t2 ON t1.stock_code=t2.stock_code
JOIN (SELECT * FROM sec_stock_history WHERE price_date='2021-03-17' AND delta_percent>=9) t3 ON t1.stock_code=t3.stock_code;

SELECT t2.delta_percent, #t3.delta_percent, 
	t1.*
FROM (SELECT * FROM sec_stock_history WHERE price_date='2021-04-09' AND delta_percent>=9) t1
JOIN (SELECT * FROM sec_stock_history WHERE price_date='2021-04-08' AND delta_percent>=9) t2 ON t1.stock_code=t2.stock_code
JOIN (SELECT * FROM sec_stock_history WHERE price_date='2021-04-07' AND delta_percent>=9) t3 ON t1.stock_code=t3.stock_code;

SELECT t2.delta_percent, #t3.delta_percent, 
	t1.*
FROM (SELECT * FROM sec_stock_history WHERE price_date='2021-04-09' AND delta_percent<-9) t1
JOIN (SELECT * FROM sec_stock_history WHERE price_date='2021-04-08' AND delta_percent<-9) t2 ON t1.stock_code=t2.stock_code
JOIN (SELECT * FROM sec_stock_history WHERE price_date='2021-04-07' AND delta_percent<-9) t3 ON t1.stock_code=t3.stock_code;


SELECT DATE_FORMAT(price_date, '%Y%m%d') price_date, stock_code, stock_name, close_price, high_price, low_price, open_price, IFNULL(ROUND(delta_percent,0), 0) delta_percent
FROM sec_stock_history 
WHERE stock_code='1000011' AND price_date>=DATE_SUB(CURDATE(), INTERVAL 2 YEAR) 
ORDER BY price_date DESC;
#菲达环保 0600526
#深物业A 1000011

DESC sec_stock_history;
DESC sec_stock_continuity;

TRUNCATE TABLE sec_stock_continuity;

SHOW PROCESSLIST;

SELECT CONCAT(sh_sz_indicator, stock_code) stock_code, stock_name
FROM sec_stock 
WHERE region_code='CN' AND stock_type='股票' AND price_time>='2021-04-09'
ORDER BY stock_code;
SELECT * FROM sec_stock_history WHERE price_date>='2021-04-09';

SELECT * FROM sec_stock_continuity 
#where end_date>='2021-04-08'
ORDER BY up9_days DESC
;

SELECT * FROM sec_stock_continuity 
WHERE end_date>='2021-03-01' AND up9_days>=3
ORDER BY end_date DESC
;

SELECT * FROM sec_stock_continuity 
WHERE end_date>='2021-03-01' AND up9_days>=3
ORDER BY stock_name, end_date DESC
;


SELECT t.*, DATE_FORMAT(start_date, '%w') start_day, DATE_FORMAT(end_date, '%w') end_day FROM sec_stock_continuity t
ORDER BY up9_days DESC
;


SELECT * FROM sec_stock WHERE percent>=0.08 AND price_time>=CURDATE() ORDER BY percent DESC;

CREATE TABLE sec_stock_continuity_0411 AS SELECT * FROM sec_stock_continuity;
SELECT * FROM sec_stock_continuity_0411;
TRUNCATE TABLE sec_stock_continuity;

SELECT * FROM sec_stock WHERE stock_code IN ('002053','000966','002639','600526','002762');
SELECT * FROM sec_stock WHERE stock_code IN (
	'603759','605136','003040','605199','400084','003039', '605188');
	

SELECT * FROM sec_stock WHERE go_market_date IS NOT NULL ORDER BY go_market_date DESC;

#@#查找当前与前一天大涨的股票，仅更新这部分股票的历史数据，以节省更新历史数据的时间
SELECT stock_code, stock_name FROM sec_stock_history WHERE price_date='2021-04-09' AND delta_percent>=9
UNION
SELECT CONCAT(sh_sz_indicator, stock_code) stock_code, stock_name FROM sec_stock WHERE price_time>='2021-04-12' AND percent >=0.09
ORDER BY stock_code
;

#@#哪些大涨股没有历史股价数据
SELECT CONCAT('http://quotes.money.163.com/service/chddata.html?code=', CONCAT(sh_sz_indicator, stock_code)) download_history, 
	CONCAT(sh_sz_indicator, stock_code) history_code, s.*
FROM sec_stock s WHERE price_time>='2021-04-12' AND percent >=0.09
	AND CONCAT(sh_sz_indicator, stock_code) NOT IN (
	SELECT stock_code FROM sec_stock_history WHERE price_date>=DATE_SUB(CURDATE(), INTERVAL 10 DAY)
	)
ORDER BY go_market_date;

#@#哪些大跌股没有历史股价数据
SELECT CONCAT('http://quotes.money.163.com/service/chddata.html?code=', CONCAT(sh_sz_indicator, stock_code)) download_history, 
	CONCAT(sh_sz_indicator, stock_code) history_code, s.*
FROM sec_stock s WHERE price_time>='2021-04-12' AND percent <=-0.09
	AND CONCAT(sh_sz_indicator, stock_code) NOT IN (
	SELECT stock_code FROM sec_stock_history WHERE price_date>=DATE_SUB(CURDATE(), INTERVAL 10 DAY)
	)
ORDER BY go_market_date;

#@#哪些大跌股当天有跌停或大跌，以至吓倒散户
SELECT CONCAT('http://quotes.money.163.com/service/chddata.html?code=', CONCAT(sh_sz_indicator, stock_code)) download_history, 
	ROUND(((high_price - low_price)/yest_close*100),0) wave, 
	CONCAT(sh_sz_indicator, stock_code) history_code, s.*
FROM sec_stock s WHERE price_time>='2021-04-12' AND ((high_price - low_price)/yest_close) >=0.19

ORDER BY go_market_date;
	AND CONCAT(sh_sz_indicator, stock_code) NOT IN (
	SELECT stock_code FROM sec_stock_history WHERE price_date>=DATE_SUB(CURDATE(), INTERVAL 10 DAY)
	)
SELECT * FROM sec_stock_history WHERE stock_code='1003032';

#@#前后两天对比，看哪些股预测大涨是成功的，哪些是失败的
SELECT * 
FROM (
	SELECT * FROM sec_stock_continuity WHERE END_DATE='2021-04-12' AND up9_days>=2 ORDER BY up9_days DESC) t1
LEFT JOIN (
	SELECT * FROM sec_stock_continuity WHERE END_DATE='2021-04-09' AND up9_days>=2 ORDER BY up9_days DESC) t2
ON t1.stock_code=t2.stock_code
ORDER BY t1.up9_days DESC;

UPDATE sec_stock_continuity c
SET open_price = (
	SELECT open_price
	FROM sec_stock_history h
	WHERE h.price_date=c.start_date AND c.stock_code=h.stock_code
)
WHERE EXISTS(
	SELECT 1
	FROM sec_stock_history h
	WHERE h.price_date=c.start_date AND c.stock_code=h.stock_code
);

UPDATE sec_stock_continuity c
SET close_price = (
	SELECT close_price
	FROM sec_stock_history h
	WHERE h.price_date=c.end_date AND c.stock_code=h.stock_code
)
WHERE EXISTS(
	SELECT 1
	FROM sec_stock_history h
	WHERE h.price_date=c.end_date AND c.stock_code=h.stock_code
);

UPDATE sec_stock_continuity c
SET real_percent = ROUND((close_price - open_price)/open_price*100, 0);

        SELECT * 
        FROM sec_stock t 
        WHERE region_code='CN' AND section_name<>'中证' 
		AND EXISTS(
			SELECT 1 
			FROM sec_stock_continuity c 
			WHERE end_date>=DATE_SUB(CURDATE(), INTERVAL 4 DAY) 
				AND c.up9_days >=2
				AND c.stock_code = CONCAT(t.sh_sz_indicator, t.stock_code)
		)
        ORDER BY fund_welcome_cnt DESC;
/*
        SELECT history_map_code stock_code 
        FROM sec_stock t 
        WHERE region_code='CN' AND section_name<>'中证' 
		and exists(
			select 1 
			from sec_stock_continuity c 
			where end_date>=date_sub(curdate(), interval 4 day) 
				and c.up9_days >=2
				and c.stock_code = t.history_map_code
		)
        ORDER BY fund_welcome_cnt DESC;
        
        SELECT CONCAT(sh_sz_indicator, stock_code) stock_code 
        FROM sec_stock t 
        WHERE region_code='CN' AND section_name<>'中证' 
		AND history_map_code in (
			SELECT stock_code 
			FROM sec_stock_continuity c 
			WHERE end_date>=DATE_SUB(CURDATE(), INTERVAL 4 DAY) 
				AND c.up9_days >=2
				#AND c.stock_code = CONCAT(t.sh_sz_indicator, t.stock_code)
		);
		
	select * from sec_stock;
	alter table sec_stock add history_map_code varchar(20);
	create unique index idx_history_map_code on sec_stock(history_map_code);
	update sec_stock t set history_map_code=CONCAT(t.sh_sz_indicator, t.stock_code);
	alter table sec_stock drop index idx_history_map_code;
	alter table sec_stock drop column history_map_code;
*/

/*
	analyze_type, stock_code, stock_name, latest_price, delta_days, total_delta_percent, delta_start_date, delta_end_date, low_price, high_price
	analyze_type: 大涨, 大跌, 悄涨，悄跌
	key: analyze_type, stock_code, delta_start_date
*/

#@#股票连续性大跌分析，发现股票中当前正在大幅下跌的股，得到入口机会。
#连接性分析中，如果几天连续跌停中有一个是没有跌停甚至是上涨了，应当忽略这天的数据。

#@#股票连续性悄涨分析，发现股票中一直在稳步上涨，但每次悄悄上涨、幅度不明显、不易被发现的股票
#偶尔有个别天下跌应当要忽略，因为它不影响整体上涨的趋势。


#@#股票连续性悄跌分析，发现股票中一直在稳步下跌，但每次悄悄下涨、幅度不明显、不易被发现的股票
#偶尔有个别天上涨应当要忽略，因为它不影响整体下跌的趋势。



#@#使用历史数据对规则进行检验准确率。


SELECT * FROM sec_fund_stock_mapping WHERE stock_code = '6005191';
SELECT * FROM sec_fund_stock_mapping;
SELECT * FROM sec_stock WHERE stock_code IN (
	'600519', '600031'
	'002179', '000858', '002340', '300014'
);
SELECT DISTINCT SUBSTR(stock_code, -1, 1) FROM sec_fund_stock_mapping WHERE LENGTH(stock_code)=7;
SELECT * FROM sec_fund_stock_mapping WHERE LENGTH(stock_code)=7 AND SUBSTR(stock_code, -1, 1)='W';
#000858 五粮液没有分类信息，需查明原因并补全

#@#检查投资于美股或港股的CN基金
SELECT * FROM (
	SELECT stock_code, COUNT(1) cnt, CASE WHEN SUBSTR(stock_code, -1, 1)='7' THEN SUBSTR(stock_code, 1, LENGTH(stock_code)-1) ELSE stock_code END stock_code2
	FROM sec_fund_stock_mapping
	WHERE LENGTH(stock_code)!=7
	GROUP BY stock_code
) t1
LEFT JOIN sec_stock s ON t1.stock_code2 = s.stock_code;

#@#检查有哪些股是基金的前10重仓股，但在股票表里没有信息
SELECT t1.* 
FROM (
	SELECT stock_code, COUNT(1) cnt
	FROM sec_fund_stock_mapping
	GROUP BY stock_code
) t1
LEFT JOIN (SELECT #t.*, 
	CASE 
		WHEN stock_code IN ('002685', '002855', '002885', '002915', 'AZN', 'RIO') THEN stock_code#特殊个例特殊处理
		WHEN subsection_name LIKE '沪%' OR (region_code='CN' AND section_name IS NULL) THEN CONCAT(stock_code, '1')#沪股后补1
		WHEN subsection_name LIKE '深%' OR subsection_name='创业板' THEN CONCAT(stock_code, '2')#深股后补2
		WHEN region_code='US' THEN CONCAT(stock_code, '7')#美股后补7
		ELSE stock_code
	END AS stock_code2
	FROM sec_stock t) s ON t1.stock_code = s.stock_code2
WHERE s.stock_code2 IS NULL
ORDER BY cnt DESC;

#@#对关联不到股票表的基金前10重仓股进行原因检查
SELECT * 
FROM (
	SELECT t1.stock_code, cnt,
		CASE WHEN LENGTH(t1.stock_code)=7 THEN SUBSTR(t1.stock_code, 1, LENGTH(t1.stock_code)-1) 
			ELSE t1.stock_code 
		END stock_code2
	FROM (
		SELECT stock_code, COUNT(1) cnt
		FROM sec_fund_stock_mapping
		GROUP BY stock_code
	) t1
	LEFT JOIN (SELECT #t.*, 
		CASE 
			WHEN stock_code IN ('002685', '002855', '002885', '002915', 'AZN', 'RIO') THEN stock_code
			WHEN subsection_name LIKE '沪%' OR (region_code='CN' AND section_name IS NULL) THEN CONCAT(stock_code, '1')
			WHEN subsection_name LIKE '深%' OR subsection_name='创业板' THEN CONCAT(stock_code, '2')
			WHEN region_code='US' THEN CONCAT(stock_code, '7')
			ELSE stock_code
		END AS stock_code2
		FROM sec_stock t) s ON t1.stock_code = s.stock_code2
	WHERE s.stock_code2 IS NULL
	ORDER BY cnt DESC
) s2 LEFT JOIN sec_stock s1 ON s1.stock_code = s2.stock_code2
ORDER BY stock_code2;

SELECT * FROM sec_stock WHERE region_code='CN' AND section_name IS NULL;


SELECT * FROM sec_stock WHERE region_code='US' AND stock_code LIKE 'AAPL%';

SELECT * FROM sec_stock ORDER BY fund_welcome_cnt DESC;
SELECT * FROM sec_stock ORDER BY fund_welcome_percent DESC;

SELECT ROUND((fund_welcome_cnt/2394)*0.5 + fund_welcome_percent/15.23*0.5,3) fund_welcome_weight, t.* 
FROM sec_stock t 
ORDER BY fund_welcome_weight DESC;

/*方式二：计算半查询，结果无保存，性能低*/
SELECT * 
FROM (
	SELECT stock_code, 
		COUNT(1) fund_welcome_cnt, 
		ROUND(SUM(fund_percent),2) fund_welcome_percent
	FROM sec_fund_stock_mapping
	GROUP BY stock_code
) m 
LEFT JOIN sec_stock s ON m.stock_code=s.stock_code
/* where s.stock_code in (
	SELECT substr(stock_code, 1,6) FROM sec_fund_stock_mapping WHERE fund_code ='003834'  
	#400015-东方新能源 002084-新华鑫动力 161725-招商白酒 001475-易方达国防军工
	#003834-华夏能源革新
) */
ORDER BY m.fund_welcome_percent DESC;
#ORDER BY m.fund_welcome_cnt DESC;

/* Supplement stock which missing before */
SELECT * FROM sec_stock WHERE stock_code IN (
'000858','300760','688169','003028','003029','002985',
'300782','688111','603267','300759','300841','688020',
'300772','688116','603915','000860','300558','688188',
'002867','688008','300674','300529','688002','688011',
'300776','600989','300791','300896','688200','002984',
'688017','603613','002979','300775','603236','688093',
'688127','688560','601995','688023','688561','603392',
'002967','002977','601865','603185','688012','688019',
'688063','688536','603187','688388','003012','688036',
'603317','002960','601615','688122','300751','300769',
'688390','688599','300792','300777','688126','300999',
'605338','605376','688363','688550','688981','300785',
'688202','300860','688099','003030','003031','605155',
'605179','605277','688567','300866','300763','300806',
'002970','601658','603290','688029','688208','688160',
'603087','300773','688580','688005','688006','688030',
'688368','688139','688396','002959','002968','688299',
'688300','689009','603195','688526','002989','300880',
'688336','688418','688488','688500','688508','688521',
'688586','688088','688321','001914','688408','601236',
'300767','300861','688301','601916','688617','002948',
'601456','688686','688698','688037','003816','688339',
'688016','601816','000876','605111','605288','688268','003001','003006','003009','003010','688015','688510','688598','300770','603068','688166','688198','605358','688595','688608','688050','300837','300888','688356','688055','688289','300850','300762','688311','300832','688369','002946','603489','688133','688058','688258','688777','601319','605123','300766','300802','000877','688065','002939','605009','002982','002947','688516','002942','300577','300788','603700','688158','300825','688788','002993','300867','300872','300919','300926','688679','688185','002958','688180','601598','601975','688021','688157','688196','300768','688009','300869','300877','300878','688165','688528','688596','688179','688298','300887','300827','300815','688519','688360','002949','300816','603786','601162','600928','300823','688333','601860','688557','688678','603927','688377','000848','000851','300900','300901','300910','688136','688513','688529','688590','688051','001872','300868','300870','300898','300761','603992','688568','002978','601827','688027','603995','688578','000869','688095','688277','605068','688066','688309','300758','002940'
) ORDER BY fund_welcome_cnt DESC;

/* Check which stocks are missing price */
SELECT CONCAT(CASE WHEN SUBSTRING(stock_code, 1, 1) IN ('0','3') THEN 1 ELSE 0 END, stock_code) stock_code, t.*
FROM sec_stock t 
WHERE region_code='CN' AND price_time IS NULL
ORDER BY fund_welcome_cnt DESC;

#@# 国内股票分类
SELECT prefix, postfix, COUNT(1) FROM (
	SELECT SUBSTR(stock_code, 1,3) prefix, SUBSTR(stock_code, 7, 1) postfix FROM (
		SELECT DISTINCT region_stock_code AS stock_code
		FROM sec_fund_stock_mapping
		WHERE LENGTH(region_stock_code)=7
	) a
) b GROUP BY prefix, postfix
ORDER BY 2,1;

#@#查看各股票受基金的欢迎程度
SELECT CONCAT(FORMAT((percent*100),2),'%') percent2, t.*
FROM sec_stock t
WHERE fund_welcome_cnt>=50 #and industry='食品饮料'
ORDER BY percent DESC;
ORDER BY fund_welcome_cnt DESC;

#@#查看前100优质股的行业分布，从而知道当年的行业方向
SELECT industry, 
	COUNT(1) stock_cnt, 
	SUM(fund_welcome_cnt) fund_welcome_cnt, 
	ROUND(SUM(fund_welcome_percent),2) fund_welcome_percent
FROM (
	SELECT @rowno:=@rowno+1 AS rowno, r.* 
	FROM sec_stock r ,(SELECT @rowno:=0) t 
	ORDER BY fund_welcome_cnt DESC
) a 
WHERE rowno<=100 
GROUP BY industry 
ORDER BY 3 DESC;

#@#查看前100优质股的行业分布，从而知道当年的地理集中地，反映地区经济的发展优劣
SELECT province, COUNT(1) 
FROM (
	SELECT @rowno:=@rowno+1 AS rowno, r.* 
	FROM sec_stock r ,(SELECT @rowno:=0) t 
	ORDER BY fund_welcome_cnt DESC
) a 
WHERE rowno<=100 
GROUP BY province 
ORDER BY 2 DESC;

#@#选前8白酒
SELECT stock_code , s.*
FROM (
	SELECT @rowno:=@rowno+1 AS rn, a.* FROM (
		SELECT * FROM sec_stock WHERE industry LIKE '%饮%' ORDER BY now_price DESC
	) a,(SELECT @rowno:=0) t
) s WHERE rn<=8;

#@#查询前100受基金欢迎的股票，并生成带region的stock code，以便用于更新小部分股票最新数据-为提高数据分析效率
SELECT CONCAT(CASE WHEN SUBSTRING(stock_code, 1, 1) IN ('0','3') THEN 1 ELSE 0 END, stock_code) stock_code
	,t.*
FROM sec_stock t 
WHERE fund_welcome_cnt>=100 
ORDER BY fund_welcome_cnt DESC;

#@#依最新股票数据——查看领涨板块——确定当前主题
SELECT t.stock_type, t.industry, t.stock_cnt, 
	a10.cnt up10, a8.cnt up8, a6.cnt up6, a4.cnt up4,
	b10.cnt down10, b8.cnt down8, b6.cnt down6, b4.cnt down4
FROM (
	SELECT stock_type, industry, COUNT(1) stock_cnt
	FROM sec_stock 
	WHERE region_code='CN' AND percent IS NOT NULL AND price_time>=DATE_SUB(CURDATE(), INTERVAL 10 DAY)
		#AND percent >= 0.06
	GROUP BY stock_type, industry
) t
LEFT JOIN (
	SELECT stock_type, industry, COUNT(1) cnt
	FROM sec_stock 
	WHERE region_code='CN' AND percent IS NOT NULL AND price_time>=DATE_SUB(CURDATE(), INTERVAL 10 DAY)
		AND percent >= 0.095
	GROUP BY stock_type, industry
) a10 ON t.industry=a10.industry
LEFT JOIN (
	SELECT stock_type, industry, COUNT(1) cnt
	FROM sec_stock 
	WHERE region_code='CN' AND percent IS NOT NULL AND price_time>=DATE_SUB(CURDATE(), INTERVAL 10 DAY)
		AND percent BETWEEN 0.08 AND 0.094
	GROUP BY stock_type, industry
) a8 ON t.industry=a8.industry
LEFT JOIN (
	SELECT stock_type, industry, COUNT(1) cnt
	FROM sec_stock 
	WHERE region_code='CN' AND percent IS NOT NULL AND price_time>=DATE_SUB(CURDATE(), INTERVAL 10 DAY)
		AND percent BETWEEN 0.06 AND 0.079
	GROUP BY stock_type, industry
) a6 ON t.industry=a6.industry
LEFT JOIN (
	SELECT stock_type, industry, COUNT(1) cnt
	FROM sec_stock 
	WHERE region_code='CN' AND percent IS NOT NULL AND price_time>=DATE_SUB(CURDATE(), INTERVAL 10 DAY)
		AND percent BETWEEN 0.04 AND 0.059
	GROUP BY stock_type, industry
) a4 ON t.industry=a4.industry
LEFT JOIN (
	SELECT stock_type, industry, COUNT(1) cnt
	FROM sec_stock 
	WHERE region_code='CN' AND percent IS NOT NULL AND price_time>=DATE_SUB(CURDATE(), INTERVAL 10 DAY)
		AND percent <=-0.095
	GROUP BY stock_type, industry
) b10 ON t.industry=b10.industry
LEFT JOIN (
	SELECT stock_type, industry, COUNT(1) cnt
	FROM sec_stock 
	WHERE region_code='CN' AND percent IS NOT NULL AND price_time>=DATE_SUB(CURDATE(), INTERVAL 10 DAY)
		AND percent BETWEEN -0.094 AND -0.08
	GROUP BY stock_type, industry
) b8 ON t.industry=b8.industry
LEFT JOIN (
	SELECT stock_type, industry, COUNT(1) cnt
	FROM sec_stock 
	WHERE region_code='CN' AND percent IS NOT NULL AND price_time>=DATE_SUB(CURDATE(), INTERVAL 10 DAY)
		AND percent BETWEEN -0.079 AND -0.06
	GROUP BY stock_type, industry
) b6 ON t.industry=b6.industry
LEFT JOIN (
	SELECT stock_type, industry, COUNT(1) cnt
	FROM sec_stock 
	WHERE region_code='CN' AND percent IS NOT NULL AND price_time>=DATE_SUB(CURDATE(), INTERVAL 10 DAY)
		AND percent BETWEEN -0.059 AND -0.04
	GROUP BY stock_type, industry
) b4 ON t.industry=b4.industry
ORDER BY a10.cnt DESC, up6 DESC, up4 DESC;
#@#明细
SELECT CONCAT(ROUND(s.percent*100, 2),'%') delta, s.*
FROM sec_stock s
WHERE region_code='CN' #and percent is not null and price_time>=DATE_SUB(CURDATE(), INTERVAL 1 DAY)
	#AND percent >= 0.09
	#AND percent <= -0.06
	AND industry='汽车'
ORDER BY percent DESC;

SELECT * FROM sec_stock WHERE stock_code='605111';


SELECT COUNT(1) #CONCAT(sh_sz_indicator, stock_code) stock_code, stock_name 
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

#@#基于股票历史数据，分析每天的热点行业
#整理成存储过程，对每天进行分析，依结果数据画图
CREATE TABLE temp_sec_stock_history AS SELECT * FROM sec_stock_history WHERE price_date='2021-03-31';
ALTER TABLE temp_sec_stock_history ADD stock_type VARCHAR(30);
ALTER TABLE temp_sec_stock_history ADD industry VARCHAR(30);
CREATE INDEX idx_temp_history ON temp_sec_stock_history(stock_code);

TRUNCATE TABLE temp_sec_stock_history;

INSERT INTO temp_sec_stock_history(
	price_date, stock_code, stock_name, close_price, high_price, low_price, open_price, yest_close, 
	delta_amount, delta_percent, handover_rate, trading_volume, turnover_amount,
	stock_type, industry)
SELECT h.price_date, h.stock_code, h.stock_name, h.close_price, h.high_price, h.low_price, h.open_price, h.yest_close, 
	h.delta_amount, h.delta_percent, h.handover_rate, h.trading_volume, h.turnover_amount,
	s.stock_type, s.industry
FROM (SELECT * FROM sec_stock_history WHERE price_date='2021-04-19') h, sec_stock s
WHERE h.stock_code = CONCAT(s.sh_sz_indicator, s.stock_code);

SELECT * FROM temp_sec_stock_history t;

UPDATE temp_sec_stock_history a
SET stock_type = (SELECT stock_type FROM sec_stock b WHERE region_code='CN' AND a.stock_code = CONCAT(b.sh_sz_indicator, b.stock_code)),
    industry = (SELECT industry FROM sec_stock b WHERE region_code='CN' AND a.stock_code = CONCAT(b.sh_sz_indicator, b.stock_code))
WHERE EXISTS (SELECT 1 FROM sec_stock b WHERE region_code='CN' AND a.stock_code = CONCAT(b.sh_sz_indicator, b.stock_code));

#@#查看昨天大涨的股票今天怎么样了，是涨还是跌
SELECT ROUND(h.delta_percent,0) yest_percent, ROUND(s.percent*100,0) today_percent, s.*
FROM sec_stock s
JOIN temp_sec_stock_history h ON CONCAT(s.sh_sz_indicator, s.stock_code) = h.stock_code
#where s.industry IN ('汽车')
ORDER BY yest_percent #desc #查看昨天大涨或大跌
;

SELECT * FROM temp_sec_stock_history;

#@#依历史股票数据——查看领涨板块——确定当前主题
SELECT price_date, t.stock_type, t.industry, t.stock_cnt, 
	a10.cnt up10, a8.cnt up8, a6.cnt up6, a4.cnt up4, a0.cnt up0,
	b10.cnt down10, b8.cnt down8, b6.cnt down6, b4.cnt down4
FROM (
	SELECT DISTINCT price_date, stock_type, IFNULL(industry,'*') industry, COUNT(1) stock_cnt
	FROM temp_sec_stock_history 
	#WHERE delta_percent >= 0.06
	GROUP BY price_date, stock_type, industry
) t
LEFT JOIN (
	SELECT stock_type, industry, COUNT(1) cnt
	FROM temp_sec_stock_history 
	WHERE delta_percent >= 0.095*100
	GROUP BY stock_type, industry
) a10 ON t.industry=a10.industry
LEFT JOIN (
	SELECT stock_type, industry, COUNT(1) cnt
	FROM temp_sec_stock_history 
	WHERE delta_percent BETWEEN 0.08*100 AND 0.094*100
	GROUP BY stock_type, industry
) a8 ON t.industry=a8.industry
LEFT JOIN (
	SELECT stock_type, industry, COUNT(1) cnt
	FROM temp_sec_stock_history 
	WHERE delta_percent BETWEEN 0.06*100 AND 0.079*100
	GROUP BY stock_type, industry
) a6 ON t.industry=a6.industry
LEFT JOIN (
	SELECT stock_type, industry, COUNT(1) cnt
	FROM temp_sec_stock_history 
	WHERE delta_percent BETWEEN 0.04*100 AND 0.059*100
	GROUP BY stock_type, industry
) a4 ON t.industry=a4.industry
LEFT JOIN (
	SELECT stock_type, industry, COUNT(1) cnt
	FROM temp_sec_stock_history 
	WHERE delta_percent BETWEEN -0.039*100 AND 0.039*100
	GROUP BY stock_type, industry
) a0 ON t.industry=a0.industry
LEFT JOIN (
	SELECT stock_type, industry, COUNT(1) cnt
	FROM temp_sec_stock_history 
	WHERE delta_percent <=-0.095*100
	GROUP BY stock_type, industry
) b10 ON t.industry=b10.industry
LEFT JOIN (
	SELECT stock_type, industry, COUNT(1) cnt
	FROM temp_sec_stock_history 
	WHERE delta_percent BETWEEN -0.094*100 AND -0.08*100
	GROUP BY stock_type, industry
) b8 ON t.industry=b8.industry
LEFT JOIN (
	SELECT stock_type, industry, COUNT(1) cnt
	FROM temp_sec_stock_history 
	WHERE delta_percent BETWEEN -0.079*100 AND -0.06*100
	GROUP BY stock_type, industry
) b6 ON t.industry=b6.industry
LEFT JOIN (
	SELECT stock_type, industry, COUNT(1) cnt
	FROM temp_sec_stock_history 
	WHERE delta_percent BETWEEN -0.059*100 AND -0.04*100
	GROUP BY stock_type, industry
) b4 ON t.industry=b4.industry
ORDER BY a10.cnt DESC, up6 DESC, up4 DESC;

#@#明细
SELECT CONCAT(ROUND(s.delta_percent, 2),'%') delta, s.*
FROM temp_sec_stock_history s
WHERE 1=1
	AND delta_percent >= 0.09
	#AND percent <= -0.06
	AND industry='电子'
ORDER BY delta_percent DESC;

DROP PROCEDURE IF EXISTS analyze_stock_hot;

DELIMITER $$
CREATE PROCEDURE analyze_stock_hot(IN p_price_date VARCHAR(10))
BEGIN
	SET @sql = 'TRUNCATE TABLE temp_sec_stock_history';
	PREPARE trunc_table FROM @sql;
	EXECUTE trunc_table;
	DEALLOCATE PREPARE trunc_table;

	INSERT INTO temp_sec_stock_history(
		price_date, stock_code, stock_name, close_price, high_price, low_price, open_price, yest_close, 
		delta_amount, delta_percent, handover_rate, trading_volume, turnover_amount,
		stock_type, industry)
	SELECT h.price_date, h.stock_code, h.stock_name, h.close_price, h.high_price, h.low_price, h.open_price, h.yest_close, 
		h.delta_amount, h.delta_percent, h.handover_rate, h.trading_volume, h.turnover_amount,
		s.stock_type, s.industry
	#FROM (SELECT * FROM sec_stock_history WHERE price_date=p_price_date) h, sec_stock s
	FROM sec_stock_history h, sec_stock s #Performance is good
	WHERE h.price_date=p_price_date AND s.region_code='CN' 
		AND h.stock_code = CONCAT(s.sh_sz_indicator, s.stock_code);
	COMMIT;
	SELECT 'abc';
	SELECT 'efg';
END$$

DELIMITER ;

CALL analyze_stock_hot('2021-03-24');

CREATE TABLE sec_stock_hot(price_date DATE);
TRUNCATE TABLE sec_stock_hot;
INSERT INTO sec_stock_hot VALUES('2021-03-21');
SELECT * FROM sec_stock_hot;
SELECT MAX(price_date) max_price_date FROM sec_stock_hot;
SELECT DATE_ADD(CURDATE(), INTERVAL 1 DAY);

DROP PROCEDURE IF EXISTS main_analyze_stock_hot;
DELIMITER $$
CREATE PROCEDURE main_analyze_stock_hot()
BEGIN
	DECLARE max_price_date DATE;
	SELECT MAX(price_date) INTO max_price_date FROM sec_stock_hot;
	WHILE max_price_date < CURDATE() DO
		SELECT MAX(price_date) INTO max_price_date FROM sec_stock_hot;
		INSERT INTO sec_stock_hot(price_date)
		VALUES (DATE_ADD(max_price_date, INTERVAL 1 DAY));
	END WHILE;
	COMMIT;
END$$
DELIMITER ;

CALL main_analyze_stock_hot();

SELECT * FROM temp_sec_stock_history;
SELECT COUNT(1) FROM sec_stock WHERE region_code='CN';

SELECT * FROM information_schema.columns WHERE table_name LIKE 'sec_%';

SELECT * FROM sec_fund_trend;
SELECT * FROM sec_stock_history WHERE stock_code = '0600519' AND price_date >=DATE_SUB(CURDATE(), INTERVAL 100 DAY) ORDER BY 1 DESC ;

SELECT DATE_SUB(CURDATE(), INTERVAL 100 DAY);

SELECT * FROM sec_fund_stock_mapping WHERE stock_code LIKE '%600872%';
SELECT * FROM sec_stock WHERE stock_code='600872' OR stock_name LIKE '%证券' ORDER BY fund_welcome_cnt DESC;
SELECT * FROM sec_stock WHERE price_time>CURDATE() AND percent>0.09 ORDER BY price_time DESC;
SELECT #* 
	section_value, COUNT(1)
FROM sec_stock s, sec_stock_section_pivot p 
WHERE p.section_name='concept' AND s.stock_code=p.stock_code AND price_time>CURDATE() AND percent>0.09 
GROUP BY section_value
ORDER BY 2 DESC;


SELECT * FROM sec_stock WHERE stock_name IN ('中远海发','永茂泰','中钢国际','高新兴','极米科技','深水海纳','','','','');
SELECT * FROM sec_stock WHERE price_time>CURDATE() AND percent<-0.09 ORDER BY price_time DESC;
SELECT CURDATE();

SELECT * FROM sec_stock_history;

#@#股票当天价格天地走势分析
SELECT * FROM sec_stock 
WHERE price_time >=DATE_SUB(CURDATE(), INTERVAL 1 DAY) AND stock_type='股票' 
	#now_price=0 #退市股
	#now_price<>0 and open_price=0 #停牌股
	
	AND open_price<>0 #可交易股
		#and percent>=0.09 #当天以停涨价收盘
		#and low_price>=yest_close*1.09 #当天是一字板涨停
		#and low_price<=yest_close*(1-0.09)#当天有跌停过
		#and high_price<=yest_close*(1-0.09) ##当天是一字板跌停
		AND low_price<=yest_close*(1-0.09) AND high_price>=yest_close*(1+0.09) #当天有过地天板
		#and (high_price - low_price)/yest_close*100>=18 and (high_price - low_price)/yest_close*100<=22 #里面可能会包含科创板，不便于找地天板
;

#@#股票历史价格天地走势分析
SELECT * FROM sec_stock_history 
WHERE price_date = '2021-04-12'
	#now_price=0 #退市股
	#now_price<>0 and open_price=0 #停牌股
	
	#AND open_price<>0 #可交易股
		AND delta_percent>=9 #当天以停涨价收盘
		#and low_price>=yest_close*(1+0.09) #当天是一字板涨停
		#and low_price<=yest_close*(1-0.09)#当天有跌停过
		#and high_price<=yest_close*(1-0.09) ##当天是一字板跌停
		#AND low_price<=yest_close*(1-0.09) AND high_price>=yest_close*(1+0.09) #当天有过地天板
			#and close_price<=yest_close*(1-0.08) #地天板股票以跌停价收盘
			#AND close_price>=yest_close*(1+0.09) #地天板股票以涨停价收盘
		#and (high_price - low_price)/yest_close*100>=18 and (high_price - low_price)/yest_close*100<=22 #里面可能会包含科创板，不便于找地天板
;

#@#检验 - 以跌、涨停价买入昨天涨停股
#股票在第dx-1天为涨停收盘
SELECT * FROM sec_stock_history WHERE price_date = '2021-04-12' AND delta_percent>=9;
#对这些股在第dx天以跌停价等待买入（假设股票一有跌停就买入成功），看有多少股票去到过跌停价可以买入成功，有多少个股是亏损，亏多少，每个股的收益和平均收益各是多少，
#假设：dx日买入的股票，在dx+1日都可买出，不存在跌停导致买不出去。
SELECT analyze_type, buy_date, 
	ROUND(AVG(earn_percent),0) avg_earn_percent, ROUND(AVG(earn_percent2),0) avg_earn_percent2, 
	COUNT(1) total_invest, COUNT(gain_indicator) cnt_gain, COUNT(loss_indicator) cnt_loss, COUNT(no_indicator) cnt_no
FROM (
	SELECT '跌停价买入涨停股' analyze_type, '2021-04-13' buy_date, tx.stock_code, tx.stock_name, 
		txm1.delta_percent dxm1_percent, tx.delta_percent dx_percent, 
		tx.low_price buy_price, 
		txa1.high_price sell_price, ROUND((txa1.high_price-tx.low_price)/tx.low_price*100,0) earn_percent, 
		txa2.high_price sell_price2, ROUND((txa2.high_price-tx.low_price)/tx.low_price*100,0) earn_percent2, 
		CASE WHEN ROUND((txa1.high_price-tx.low_price)/tx.low_price*100,0)>0 THEN '赚' ELSE NULL END AS gain_indicator,
		CASE WHEN ROUND((txa1.high_price-tx.low_price)/tx.low_price*100,0)=0 THEN '平' ELSE NULL END AS no_indicator,
		CASE WHEN ROUND((txa1.high_price-tx.low_price)/tx.low_price*100,0)<0 THEN '亏' ELSE NULL END AS loss_indicator
		#, txm1.*
	FROM (SELECT * FROM sec_stock_history WHERE price_date = '2021-04-12' AND delta_percent>=9) txm1 #dx-1日涨停收盘的股票
	LEFT JOIN (SELECT * FROM sec_stock_history WHERE price_date = '2021-04-13' AND low_price<=yest_close*(1-0.09)) tx ON txm1.stock_code=tx.stock_code#dx日以跌停价买入
	LEFT JOIN (SELECT * FROM sec_stock_history WHERE price_date = '2021-04-14' ) txa1 ON txa1.stock_code=tx.stock_code#dx+1日可卖
	LEFT JOIN (SELECT * FROM sec_stock_history WHERE price_date = '2021-04-15' ) txa2 ON txa2.stock_code=tx.stock_code#dx+2日可卖
	WHERE txa1.stock_code IS NOT NULL
) t
;

SELECT '涨停价买入涨停股' analyze_type, '2021-04-13' buy_date, tx.stock_code, tx.stock_name, 
	txm1.delta_percent dxm1_percent, tx.delta_percent dx_percent, 
	tx.open_price buy_price, 
	txa1.high_price sell_price, ROUND((txa1.high_price-tx.open_price)/tx.open_price*100,0) earn_percent, 
	txa2.high_price sell_price2, ROUND((txa2.high_price-tx.open_price)/tx.open_price*100,0) earn_percent2, 
	CASE WHEN ROUND((txa1.high_price-tx.open_price)/tx.open_price*100,0)>0 THEN '赚' ELSE NULL END AS gain_indicator,
	CASE WHEN ROUND((txa1.high_price-tx.open_price)/tx.open_price*100,0)=0 THEN '平' ELSE NULL END AS no_indicator,
	CASE WHEN ROUND((txa1.high_price-tx.open_price)/tx.open_price*100,0)<0 THEN '亏' ELSE NULL END AS loss_indicator
	#, txm1.*
FROM (SELECT * FROM sec_stock_history WHERE price_date = '2021-04-12' AND delta_percent>=9) txm1 #dx-1日涨停收盘的股票
LEFT JOIN (SELECT * FROM sec_stock_history WHERE price_date = '2021-04-13' /*AND low_price<=yest_close*(1-0.09)*/) tx ON txm1.stock_code=tx.stock_code #dx日以跌停价买入
LEFT JOIN (SELECT * FROM sec_stock_history WHERE price_date = '2021-04-14' ) txa1 ON txa1.stock_code=tx.stock_code #dx+1日可卖
LEFT JOIN (SELECT * FROM sec_stock_history WHERE price_date = '2021-04-15' ) txa2 ON txa2.stock_code=tx.stock_code #dx+2日可卖
WHERE txa1.stock_code IS NOT NULL;

#TRUNCATE TABLE sec_stock_tradeatbig;
SELECT * FROM sec_stock_history WHERE stock_code='1000761' ORDER BY price_date DESC;
SELECT * FROM sec_stock_tradeatbig WHERE stock_code='0688689'
ORDER BY earn_percent DESC;


SELECT * FROM sec_stock_tradeatbig 
#WHERE buy_price=0 OR sell_price=0 OR sell_price2=0 
#	OR (gain_indicator IS NULL AND loss_indicator IS NULL AND no_indicator IS NULL)#查出有问题的数据
#where flag is null and ((earn_percent>=40 or earn_percent<=-40 or earn_percent2>=50 OR earn_percent2<=-50) or stock_name like 'C%')
WHERE flag IS NULL AND earn_percent>=28
;

UPDATE sec_stock_tradeatbig
SET flag='错误记录'
WHERE buy_price=0 OR sell_price=0 OR sell_price2=0 
	OR (gain_indicator IS NULL AND loss_indicator IS NULL AND no_indicator IS NULL)#查出有问题的数据
;

#创业板、科创板
UPDATE sec_stock_tradeatbig
SET flag='创板'
WHERE flag IS NULL AND ((earn_percent>=40 OR earn_percent<=-40 OR earn_percent2>=50 OR earn_percent2<=-50) OR stock_name LIKE 'C%');

#注意：需要把创业板、科创板的去除。
#按天统计
SELECT analyze_type, buy_date, 
	ROUND(AVG(earn_percent),0) avg_earn_percent, ROUND(AVG(earn_percent2),0) avg_earn_percent2, 
	COUNT(1) total_invest, COUNT(gain_indicator) cnt_gain, COUNT(loss_indicator) cnt_loss, COUNT(no_indicator) cnt_no,
	ROUND(COUNT(loss_indicator)/COUNT(1)*100, 2) cnt_loss_percent
FROM sec_stock_tradeatbig
#where buy_date>='2021-02-01'
WHERE flag IS NULL#去除创业板、科创板、有问题记录
GROUP BY analyze_type, buy_date
ORDER BY buy_date DESC, analyze_type;

#按月统计
SELECT analyze_type, DATE_FORMAT(buy_date, '%Y-%m') buy_date, 
	ROUND(AVG(earn_percent),0) avg_earn_percent, ROUND(AVG(earn_percent2),0) avg_earn_percent2, 
	COUNT(1) total_invest, COUNT(gain_indicator) cnt_gain, COUNT(loss_indicator) cnt_loss, COUNT(no_indicator) cnt_no,
	ROUND(COUNT(loss_indicator)/COUNT(1)*100, 2) cnt_loss_percent
FROM sec_stock_tradeatbig
WHERE flag IS NULL#去除创业板、科创板、有问题记录
GROUP BY analyze_type, DATE_FORMAT(buy_date, '%Y-%m')
ORDER BY buy_date DESC, analyze_type;

#全部统计
#按年统计
SELECT analyze_type, #buy_date, 
	ROUND(AVG(earn_percent),0) avg_earn_percent, ROUND(AVG(earn_percent2),0) avg_earn_percent2, 
	COUNT(1) total_invest, COUNT(gain_indicator) cnt_gain, COUNT(loss_indicator) cnt_loss, COUNT(no_indicator) cnt_no,
	ROUND(COUNT(loss_indicator)/COUNT(1)*100, 2) cnt_loss_percent
FROM sec_stock_tradeatbig
#where buy_date>='2021-02-01'
WHERE flag IS NULL#去除创业板、科创板、有问题记录
GROUP BY analyze_type
;
#dx+1日已经赚到30%，是否需要再等dx+2日再卖，是否会赚到40%
SELECT analyze_type, #buy_date, 
	ROUND(AVG(earn_percent),0) avg_earn_percent, ROUND(AVG(earn_percent2),0) avg_earn_percent2, 
	COUNT(1) total_invest, COUNT(gain_indicator) cnt_gain, COUNT(loss_indicator) cnt_loss, COUNT(no_indicator) cnt_no,
	ROUND(COUNT(loss_indicator)/COUNT(1)*100, 2) cnt_loss_percent
FROM sec_stock_tradeatbig
#where buy_date>='2021-02-01'
WHERE flag IS NULL#去除创业板、科创板、有问题记录
	AND earn_percent>=28
GROUP BY analyze_type
