DROP TABLE sec_stock;
CREATE TABLE sec_stock(
	create_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	stock_code VARCHAR(20),
	stock_name VARCHAR(50),
	short_pinyin VARCHAR(25),
	company_name VARCHAR(50),
	region_code VARCHAR(20), /*CN, HK, US*/
	section_name VARCHAR(20), /*沪深A股,上证A股,深证A股,新股,中小板,创业板,科创板,沪股通,深股通,B股*/
	subsection_name VARCHAR(20),
	fund_welcome INT,
	industry VARCHAR(20),
	province VARCHAR(20),
	PRIMARY KEY(stock_code)
);
ALTER TABLE sec_stock ADD COLUMN industry VARCHAR(20);
ALTER TABLE sec_stock ADD COLUMN province VARCHAR(20);

ALTER TABLE sec_stock ADD COLUMN price_time DATETIME;
ALTER TABLE sec_stock ADD COLUMN yest_close FLOAT;
ALTER TABLE sec_stock ADD COLUMN now_price FLOAT;
ALTER TABLE sec_stock ADD COLUMN arrow VARCHAR(4);
ALTER TABLE sec_stock ADD COLUMN open_price FLOAT;
ALTER TABLE sec_stock ADD COLUMN high_price FLOAT;
ALTER TABLE sec_stock ADD COLUMN low_price FLOAT;
ALTER TABLE sec_stock ADD COLUMN volume BIGINT;
ALTER TABLE sec_stock MODIFY COLUMN volume BIGINT;
ALTER TABLE sec_stock ADD COLUMN percent FLOAT;

ALTER TABLE sec_stock ADD COLUMN fh_mark VARCHAR(20);
/*Fh analyze data - to improve performance*/
ALTER TABLE sec_stock CHANGE COLUMN fund_welcome fund_welcome_cnt INT;
ALTER TABLE sec_stock ADD COLUMN fund_welcome_percent FLOAT;

ALTER TABLE sec_stock DROP PRIMARY KEY;
ALTER TABLE sec_stock ADD COLUMN stock_type VARCHAR(30) DEFAULT '股票'; /* 股票， 指数 */
ALTER TABLE sec_stock ADD PRIMARY KEY(stock_type, stock_code);

/* 0-上证, 1-深证. To improve system performance, and simplify SQL */
ALTER TABLE sec_stock ADD COLUMN sh_sz_indicator VARCHAR(1);

/* 概念板块 - 在行业基础上再细分，以便将性质相同的放在一起分析，剔除性质不同的  */
ALTER TABLE sec_stock ADD concept_bk VARCHAR(20) AFTER industry; 

/* 基金映射列 - 以方便直接和天天基金网取下的基金重仓前10股票数据表关联 */ 
ALTER TABLE sec_stock ADD COLUMN fund_map_code VARCHAR(20) AFTER sh_sz_indicator;

/* 股票上市日期 */
ALTER TABLE sec_stock ADD go_market_date DATE;

/* DROP TABLE sec_stock_type_v; */
CREATE TABLE sec_stock_section_pivot(
	create_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	section_name VARCHAR(20),
	section_value VARCHAR(100),
	stock_code VARCHAR(20)
);
CREATE INDEX idx_sec_ss_pivot ON sec_stock_section_pivot(section_name, stock_code);

ALTER TABLE sec_stock CHANGE COLUMN region_name region_code;#报错
ALTER TABLE sec_stock CHANGE COLUMN region_name region_code VARCHAR(20);#正确
ALTER TABLE sec_stock ADD COLUMN subsection_name VARCHAR(20);
ALTER TABLE sec_stock ADD COLUMN fund_welcome INT;

#DROP TABLE sec_stock_history;
CREATE TABLE sec_stock_history(
	price_date DATE,
	stock_code VARCHAR(20),
	stock_name VARCHAR(50),
	close_price FLOAT,
	high_price FLOAT,
	low_price FLOAT,
	open_price FLOAT,
	yest_close FLOAT,
	delta_amount FLOAT, /*涨跌额*/
	delta_percent FLOAT, /*涨跌幅*/
	handover_rate FLOAT, /*换手率*/
	trading_volume BIGINT, /*成交量*/
	turnover_amount BIGINT, /*成交额*/
	total_market_value BIGINT, /*市总值*/
	circu_market_value BIGINT, /*流通市值*/
	trading_lots INT, /*成交笔数*/
	PRIMARY KEY(price_date, stock_code)
);
ALTER TABLE sec_stock_history MODIFY COLUMN trading_volume BIGINT;

ALTER TABLE sec_stock_history DROP PRIMARY KEY;
ALTER TABLE sec_stock_history ADD PRIMARY KEY(price_date, stock_code);

#股票连续性分析
#drop table if exists sec_stock_continuity;
CREATE TABLE sec_stock_continuity(
	analyze_type VARCHAR(10),/*大涨、大跌、悄涨、悄跌*/
	stock_code VARCHAR(20),
	stock_name VARCHAR(50),
	
	trade_days INT,
	up9_days INT,
	
	sum_percent FLOAT,
	avg_percent FLOAT,
	real_percent FLOAT, /* (close_price - open_price) / open_price*/
	
	close_price FLOAT,
	high_price FLOAT,
	low_price FLOAT,
	open_price FLOAT,
	
	start_date DATE,
	end_date DATE
);

ALTER TABLE sec_stock_continuity ADD af_up9_percent FLOAT AFTER up9_days;
ALTER TABLE sec_stock_continuity ADD bf_up9_days INT AFTER trade_days;
ALTER TABLE sec_stock_continuity ADD st_up9_date DATE;


DROP TABLE IF EXISTS sec_stock_tradeatbig;
CREATE TABLE sec_stock_tradeatbig(
	analyze_type varchar(20),/*跌停价买入涨停股, 涨停价买入涨停股, 涨停价买入一字涨停股*/
	buy_date DATE,
	stock_code VARCHAR(20),
	stock_name VARCHAR(50),
	dxm1_percent FLOAT,
	dx_percent FLOAT,
	buy_price FLOAT,
	sell_price FLOAT,
	earn_percent FLOAT,
	sell_price2 FLOAT,
	earn_percent2 FLOAT,
	gain_indicator VARCHAR(1),
	loss_indicator VARCHAR(1),
	no_indicator VARCHAR(1),
	PRIMARY KEY(analyze_type, buy_date, stock_code)
);
ALTER TABLE sec_stock_tradeatbig ADD flag VARCHAR(10);/*正确、错误、科创板、创业板*/



