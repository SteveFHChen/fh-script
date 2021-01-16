
--Completed
CREATE TABLE covid_area_stat2006 AS SELECT * FROM covid_area_stat WHERE business_date >='2020-06-01 00:00:00' AND business_date<'2020-07-01 00:00:00';
CREATE TABLE covid_area_stat2007 AS SELECT * FROM covid_area_stat WHERE business_date >='2020-07-01 00:00:00' AND business_date<'2020-08-01 00:00:00';
CREATE TABLE covid_area_stat2008 AS SELECT * FROM covid_area_stat WHERE business_date >='2020-08-01 00:00:00' AND business_date<'2020-09-01 00:00:00';
CREATE TABLE covid_area_stat2009 AS SELECT * FROM covid_area_stat WHERE business_date >='2020-09-01 00:00:00' AND business_date<'2020-10-01 00:00:00';
CREATE TABLE covid_area_stat2010 AS SELECT * FROM covid_area_stat WHERE business_date >='2020-10-01 00:00:00' AND business_date<'2020-11-01 00:00:00';
CREATE TABLE covid_area_stat2011 AS SELECT * FROM covid_area_stat WHERE business_date >='2020-11-01 00:00:00' AND business_date<'2020-12-01 00:00:00';
CREATE TABLE covid_area_stat2012 AS SELECT * FROM covid_area_stat WHERE business_date >='2020-12-01 00:00:00' AND business_date<'2021-01-01 00:00:00';

--Pending action
CREATE TABLE covid_area_stat2101 AS SELECT * FROM covid_area_stat WHERE business_date >='2021-01-01 00:00:00' AND business_date<'2021-02-01 00:00:00';
CREATE TABLE covid_area_stat2102 AS SELECT * FROM covid_area_stat WHERE business_date >='2021-02-01 00:00:00' AND business_date<'2021-03-01 00:00:00';
CREATE TABLE covid_area_stat2103 AS SELECT * FROM covid_area_stat WHERE business_date >='2021-03-01 00:00:00' AND business_date<'2021-04-01 00:00:00';
CREATE TABLE covid_area_stat2104 AS SELECT * FROM covid_area_stat WHERE business_date >='2021-04-01 00:00:00' AND business_date<'2021-05-01 00:00:00';
CREATE TABLE covid_area_stat2105 AS SELECT * FROM covid_area_stat WHERE business_date >='2021-05-01 00:00:00' AND business_date<'2021-06-01 00:00:00';



