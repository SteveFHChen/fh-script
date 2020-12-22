
CREATE TABLE covid_area_stat (
  business_date timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  AREA varchar(50),
  new_confirmed int(11),
  exist_confirmed int(11),
  total_confirmed int(11),
  recruited int(11),
  death int(11)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


CREATE TABLE user_info (
  user_id int(11) NOT NULL AUTO_INCREMENT,
  user_no varchar(20) NOT NULL,
  user_name varchar(50),
  pwd varchar(20),
  photo_file varchar(100),
  PRIMARY KEY (user_id)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8;

CREATE TABLE fact_covid(
  business_date DATE, 
  AREA VARCHAR(50),
  new_confirmed INT
);

DESC covid_area_stat;
DESC fact_covid;

SELECT COUNT(1) FROM covid_area_stat;
SELECT COUNT(1) FROM fact_covid;

/**
 * Base on the stage table covid_area_stat, 
 * move the data needed by python analysis into fact table.
 * So that python only focus on data analysis, instead of pre-process data.
 */

CREATE TABLE fact_covid_bak1219 AS SELECT * FROM fact_covid;
SELECT * FROM fact_covid_bak1219;
SELECT COUNT(1) FROM fact_covid_bak1219;

TRUNCATE TABLE fact_covid;

INSERT INTO fact_covid(business_date, AREA, new_confirmed)
SELECT date1 business_date, AREA, new_confirmed 
FROM (
	SELECT IF(@datex=t.date1 AND @areax=t.area,@rank:=@rank+1,@rank:=1) AS rank,
		t.*,@datex:=t.date1 datex, @areax:=t.area areax
	#select * 
	FROM 
	  (SELECT @datex:=NULL, @areax:=NULL, @rank:=0) r, 
	  (SELECT DATE_FORMAT(business_date,'%Y-%m-%d') date1, AREA, new_confirmed 
	      FROM covid_area_stat #WHERE AREA='法国' 
	      ORDER BY date1 DESC, AREA DESC, new_confirmed DESC) t
) t1 WHERE rank=1;

/**
 * Verify result, if no problem, then the backup table can be dropped.
 */
SELECT * FROM fact_covid;
SELECT COUNT(1) FROM fact_covid;

DROP TABLE fact_covid_bak1219;

/**
 * The sql which will be used in python to get data for analysis.
 */
SELECT @rn:=@rn+1 rn, t.*, DATE_FORMAT(business_date, '%m%d') biz_md
FROM fact_covid t, (SELECT @rn:=0) r 
WHERE AREA='法国'
ORDER BY business_date;

