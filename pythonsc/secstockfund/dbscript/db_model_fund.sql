CREATE TABLE sec_fund_company(
	create_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	company_code VARCHAR(10),
	company_name VARCHAR(50),
	PRIMARY KEY(company_code)
);


CREATE TABLE sec_fund(
	create_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	fund_code VARCHAR(10),
	fund_name VARCHAR(50),
	fund_type VARCHAR(20),
	short_pinyin VARCHAR(25),
	company_code VARCHAR(10),
	PRIMARY KEY(fund_code)
);
/*Real time data - latest value data - to improve performance*/
ALTER TABLE sec_fund ADD COLUMN net_date DATE;
ALTER TABLE sec_fund ADD COLUMN unit_net FLOAT;
ALTER TABLE sec_fund ADD COLUMN esti_net FLOAT;
ALTER TABLE sec_fund ADD COLUMN esti_rate FLOAT;
ALTER TABLE sec_fund ADD COLUMN gztime DATETIME;

/**/
ALTER TABLE sec_fund ADD COLUMN fund_scale FLOAT;

/*Fh mark field*/
ALTER TABLE sec_fund ADD COLUMN fh_mark VARCHAR(20);

/*Fh analyze data - to improve performance*/
ALTER TABLE sec_fund ADD lastn_top_rate FLOAT; 
ALTER TABLE sec_fund ADD lastn_top_date DATE;
ALTER TABLE sec_fund ADD lastn_low_rate FLOAT;
ALTER TABLE sec_fund ADD lastn_low_date DATE;
ALTER TABLE sec_fund ADD last_now_rate FLOAT;
ALTER TABLE sec_fund ADD last_now_date DATE;

/* 
 * To store the original fund and company mapping data, which is captured from internet
 * Then used to generate sec_stock.company_code value. 
 */
CREATE TABLE sec_ori_fund_company(
	a2z VARCHAR(1),
	company_name VARCHAR(30),
	fund_code VARCHAR(30),
	fund_name VARCHAR(40),
	PRIMARY KEY(company_name, fund_code)
);
ALTER TABLE sec_ori_fund_company ADD PRIMARY KEY pk_secori_fund_company(company_name, fund_code);

/* DROP TABLE sec_fund_trend; */
CREATE TABLE sec_fund_trend(
	net_date BIGINT UNSIGNED,
	fund_code VARCHAR(10),
	unit_net FLOAT,
	acc_net FLOAT,
	day_delta_rate FLOAT,
	PRIMARY KEY (net_date, fund_code)
);
/* 
#单位净值指的是某一天该基金的单位价值，即该基金当天的价格，
#而基金累计净值是基金成立以来每天增减量的累加；
#基金单位净值=（基金资产总值－基金负债）/基金总份额；
#累计单位净值=单位净值 + 成立以来每份累计分红派息的金额。
#日增长率 = ([T]单位净值 - [T-1]单位净值) / [T-1]单位净值
 */

/* DROP TABLE sec_fund_stock_mapping; */
CREATE TABLE sec_fund_stock_mapping(
	create_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	fund_code VARCHAR(10),
	region_stock_code VARCHAR(20),
	stock_code VARCHAR(20), /* Generate a purge stock_code column for easy to use */
	fund_percent FLOAT,
	PRIMARY KEY(fund_code, stock_code)
);

