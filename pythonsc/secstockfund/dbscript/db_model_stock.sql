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
	PRIMARY KEY(stock_code, price_date)
);
ALTER TABLE sec_stock_history MODIFY COLUMN trading_volume BIGINT;

