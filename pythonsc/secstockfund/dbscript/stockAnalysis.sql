/* 
    查看龙头股
    哪些股受基金的欢迎度高，哪些股不受基金欢迎
 */
/*方式一：先一次计算，再多次查询，结果有保存，性能高*/
UPDATE sec_stock a
SET fund_welcome_cnt=(
	SELECT fund_welcome_cnt FROM (
		SELECT stock_code, COUNT(1) fund_welcome_cnt, ROUND(SUM(fund_percent),2) fund_welcome_percent
		FROM sec_fund_stock_mapping
		GROUP BY stock_code
	) t1 
	WHERE t1.stock_code = a.stock_code
), fund_welcome_percent=(
	SELECT fund_welcome_percent FROM (
		SELECT stock_code, COUNT(1) fund_welcome_cnt, ROUND(SUM(fund_percent),2) fund_welcome_percent
		FROM sec_fund_stock_mapping
		GROUP BY stock_code
	) t1 
	WHERE t1.stock_code = a.stock_code
)
WHERE EXISTS(
	SELECT stock_code, cnt FROM (
		SELECT stock_code, COUNT(1) cnt
		FROM sec_fund_stock_mapping
		GROUP BY stock_code
	) t2 
	WHERE t2.stock_code = a.stock_code
);


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

/* 国内股票分类 */
SELECT prefix, postfix, COUNT(1) FROM (
	SELECT SUBSTR(stock_code, 1,3) prefix, SUBSTR(stock_code, 7, 1) postfix FROM (
		SELECT DISTINCT region_stock_code AS stock_code
		FROM sec_fund_stock_mapping
		WHERE LENGTH(region_stock_code)=7
	) a
) b GROUP BY prefix, postfix
ORDER BY 2,1;

/* 查看各股票受基金的欢迎程度 */
SELECT CONCAT(FORMAT((percent*100),2),'%') percent2, t.*
FROM sec_stock t
WHERE fund_welcome_cnt>=50 #and industry='食品饮料'
ORDER BY percent DESC;
ORDER BY fund_welcome_cnt DESC;

/* 查看前100优质股的行业分布，从而知道当年的行业方向 */
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

/* 查看前100优质股的行业分布，从而知道当年的地理集中地，反映地区经济的发展优劣 */
SELECT province, COUNT(1) 
FROM (
	SELECT @rowno:=@rowno+1 AS rowno, r.* 
	FROM sec_stock r ,(SELECT @rowno:=0) t 
	ORDER BY fund_welcome_cnt DESC
) a 
WHERE rowno<=100 
GROUP BY province 
ORDER BY 2 DESC;

/* 选前8白酒 */
SELECT stock_code , s.*
FROM (
	SELECT @rowno:=@rowno+1 AS rn, a.* FROM (
		SELECT * FROM sec_stock WHERE industry LIKE '%饮%' ORDER BY now_price DESC
	) a,(SELECT @rowno:=0) t
) s WHERE rn<=8;

/* 查询前100受基金欢迎的股票，并生成带region的stock code，以便用于更新小部分股票最新数据-为提高数据分析效率 */
SELECT CONCAT(CASE WHEN SUBSTRING(stock_code, 1, 1) IN ('0','3') THEN 1 ELSE 0 END, stock_code) stock_code
	,t.*
FROM sec_stock t 
WHERE fund_welcome_cnt>=100 
ORDER BY fund_welcome_cnt DESC;


