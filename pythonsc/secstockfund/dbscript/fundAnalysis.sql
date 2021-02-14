/* Check postfix first */
SELECT post, COUNT(1) FROM (
	SELECT SUBSTR(region_stock_code, 7) post
	FROM sec_fund_stock_mapping
	WHERE LENGTH(region_stock_code)=7 
) t GROUP BY post;

SELECT *
FROM sec_fund_stock_mapping 
WHERE stock_code IS NULL;
WHERE LENGTH(region_stock_code)=7 AND SUBSTR(region_stock_code, 7) IN ('1', '2');
/* Generate purge stock code for CN */
UPDATE sec_fund_stock_mapping
SET stock_code = SUBSTR(region_stock_code,1,6)
WHERE LENGTH(region_stock_code)=7 AND SUBSTR(region_stock_code, 7) IN ('1', '2');
/* Copy purge stock code for others */
UPDATE sec_fund_stock_mapping
SET stock_code = region_stock_code
WHERE stock_code IS NULL;

/* 根据基金规模选基金 */
SELECT * FROM sec_fund 
WHERE fund_scale IS NOT NULL AND fund_type NOT IN ('货币型', '定开债券', '债券型') 
ORDER BY fund_scale DESC;

/* 
  按基金公司拥有的基金数量倒排序，是否可以反映基金公司的实力
 */
SELECT *
FROM sec_fund_company c
LEFT JOIN (
	SELECT company_code, COUNT(1) fund_cnt 
	FROM sec_fund 
	GROUP BY company_code
) f ON c.company_code=f.company_code
ORDER BY fund_cnt DESC;

/* 
  了解各基金公司的基金类型比重
  一般：混合型 > 债券型、定开债券 > 股票型、股票指数
 */
SELECT *
FROM sec_fund_company c
LEFT JOIN (
	SELECT company_code, COUNT(1) company_fund_cnt 
	FROM sec_fund 
	GROUP BY company_code
) f1 ON c.company_code=f1.company_code
LEFT JOIN (
	SELECT company_code, fund_type, COUNT(1) comp_type_fund_cnt 
	FROM sec_fund 
	GROUP BY company_code, fund_type
) f2 ON c.company_code=f2.company_code
#order by company_name, fund_type;
ORDER BY company_fund_cnt DESC, company_name, comp_type_fund_cnt DESC;

/* 
    查询使用高度欢迎股多个的基金
 */
SELECT * FROM (
	SELECT m.fund_code, COUNT(1) welcome_stock_cnt
	FROM sec_fund_stock_mapping m
	LEFT JOIN sec_fund f ON m.fund_code=f.fund_code
	LEFT JOIN sec_stock s ON m.stock_code=s.stock_code
	WHERE fund_welcome_cnt>=680
	GROUP BY m.fund_code HAVING COUNT(1)>=8
	ORDER BY 2 DESC, 1
) a
LEFT JOIN sec_fund b ON a.fund_code=b.fund_code
WHERE fund_scale > 50
ORDER BY 2 DESC, fund_scale DESC;

/*
  
 */
UPDATE sec_fund f
SET fh_mark = (
	SELECT CONCAT('+龙头', cnt)
	FROM (
		SELECT m.fund_code, COUNT(1) cnt
		FROM sec_fund_stock_mapping m
		LEFT JOIN sec_fund f ON m.fund_code=f.fund_code
		LEFT JOIN sec_stock s ON m.stock_code=s.stock_code
		WHERE fund_welcome_cnt>=680
		GROUP BY m.fund_code HAVING COUNT(1)>=8
	) t WHERE f.fund_code=t.fund_code
)
WHERE EXISTS (
	SELECT 1
	FROM (
		SELECT m.fund_code, COUNT(1) cnt
		FROM sec_fund_stock_mapping m
		LEFT JOIN sec_fund f ON m.fund_code=f.fund_code
		LEFT JOIN sec_stock s ON m.stock_code=s.stock_code
		WHERE fund_welcome_cnt>=680
		GROUP BY m.fund_code HAVING COUNT(1)>=8
	) t WHERE f.fund_code=t.fund_code
);

/* 查找前排优质基金画曲线 */
SELECT * FROM (
	SELECT @rownum:=@rownum+1 AS rownum, t.* FROM sec_fund t, (SELECT @rownum:=0) k
	WHERE fh_mark IS NOT NULL AND net_date IS NOT NULL
	ORDER BY fund_scale DESC
) a WHERE rownum<=20 ORDER BY rownum;

SELECT * FROM (
	SELECT @rownum:=@rownum+1 rownum, a.*
	FROM sec_fund a, (SELECT @rownum:=0) b
	WHERE fh_mark IS NOT NULL AND fund_type NOT IN ('ETF-场内', '债券型') 
	ORDER BY fund_scale DESC
) t WHERE rownum<=10;

/* 根据基金最近波动范围和当前值选基金入口 */
SELECT * FROM (
	SELECT ROUND(lastn_top_rate-lastn_low_rate,3) fudu, 
	       ROUND(lastn_top_rate - last_now_rate,3) godown,
		f.* 
	FROM sec_fund f
) t
WHERE fund_scale>=100 AND fudu>10 AND godown>13
#order by fudu desc, godown desc;
ORDER BY godown DESC, fudu DESC;


/* 查看基金的最近数据更新到几日，是否是最新数据 */
SELECT FROM_UNIXTIME(net_date/1000) net_date, COUNT(1) cnt 
FROM (
	SELECT  IF(@fundcode=a.fund_code, @rank:=@rank+1, @rank:=1) rank,
		a.*, @fundcode:=a.fund_code fundcode
	FROM (
		SELECT * FROM sec_fund_trend a 
		WHERE net_date>UNIX_TIMESTAMP('2021-01-20')*1000
		ORDER BY fund_code, net_date DESC
	) a, (SELECT @fundcode:=NULL, @rank:=0) r
) t WHERE rank = 1 
	AND fundcode NOT IN (
		SELECT fund_code FROM sec_fund 
		WHERE fund_type IN ('货币型') OR fund_type LIKE '%债券%'
	)
GROUP BY net_date
ORDER BY 1 DESC;

SELECT fund_code, MAX(net_date) max_net_date, FROM_UNIXTIME(MAX(t.net_date)/1000, '%Y-%m-%d') net_date2 
#select FROM_UNIXTIME(t.net_date/1000) net_date2, t.*
FROM sec_fund_trend t
WHERE net_date >=UNIX_TIMESTAMP('2021-02-1')*1000 AND net_date<> UNIX_TIMESTAMP('2021-02-09')*1000 #To improve performance
	#and fund_code in ('161725','110003') ;
GROUP BY fund_code;

/* Check some funds trend update status */
SELECT fund_code, COUNT(1) cnt FROM (
	SELECT FROM_UNIXTIME(net_date/1000) net_date2, t.* FROM (
		SELECT  IF(@fundcode=a.fund_code, @rank:=@rank+1, @rank:=1) rank,
			a.*, @fundcode:=a.fund_code fundcode
		FROM (
			SELECT * FROM sec_fund_trend a 
			WHERE net_date>UNIX_TIMESTAMP('2021-01-20')*1000
			ORDER BY fund_code, net_date DESC
		) a, (SELECT @fundcode:=NULL, @rank:=0) r
	) t WHERE rank <= 20 AND fund_code IN (
		'511880','005827','161725','510050','510300',
		'110011','260108','512880','510500','163402'
	)
	ORDER BY fund_code, rank
) t2 GROUP BY fund_code ORDER BY fund_code;

/* 根据主题选优质基金 */
SELECT * 
FROM sec_fund
WHERE fund_name LIKE '%白酒%' OR fund_name LIKE '%能源%'
ORDER BY fund_scale DESC;

/* 查找投资了腾讯控股的基金，并按基金规则排序 */
SELECT * 
FROM sec_fund_stock_mapping m
JOIN sec_fund f ON m.fund_code=f.fund_code
WHERE m.stock_code LIKE '%00700%' 
ORDER BY fund_scale DESC;

/* 选基金投入点 */
SELECT * 
FROM sec_fund 
WHERE esti_rate>=0.2 OR esti_rate<=-0.2 
	AND fund_code IN (
		SELECT DISTINCT fund_code
		FROM sec_fund_stock_mapping m 
		JOIN sec_stock s ON m.stock_code=s.stock_code
		WHERE fund_welcome_cnt>=1000
	)
ORDER BY esti_rate DESC;

/* 检查哪些基金没有归属基金公司 */
SELECT * 
FROM sec_fund a 
LEFT JOIN sec_ori_fund_company b ON a.fund_code=b.fund_code
WHERE b.fund_code IS NULL;

SELECT * FROM sec_fund
WHERE company_code IS NULL;

SELECT a2z, company_name, COUNT(1) 
FROM sec_ori_fund_company 
GROUP BY a2z, company_name 
ORDER BY 1;

