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
ALTER TABLE sec_stock ADD COLUMN volume INT;
ALTER TABLE sec_stock ADD COLUMN percent FLOAT;

ALTER TABLE sec_stock ADD COLUMN fh_mark VARCHAR(20);
/*Fh analyze data - to improve performance*/
ALTER TABLE sec_stock CHANGE COLUMN fund_welcome fund_welcome_cnt INT;
ALTER TABLE sec_stock ADD COLUMN fund_welcome_percent FLOAT;


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


