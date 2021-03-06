/* DB model */
#DROP TABLE fhmoinfo;
CREATE TABLE fhmoinfo(
	create_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	title VARCHAR(100),
	keyname VARCHAR(30),
	url VARCHAR(200),
	img VARCHAR(200),
	STATUS VARCHAR(30) NOT NULL DEFAULT 'INIT', /*INIT, CAPTURE, FINISHED*/
	PRIMARY KEY(title)
);

ALTER TABLE fhmoinfo ADD capture_date TIMESTAMP;
ALTER TABLE fhmoinfo ADD skip_capture VARCHAR(30);

CREATE TABLE fhmotitle(
	title VARCHAR(100),
	person VARCHAR(50)
);

#根据所有mo title生成key，并将title和key mapping存入result
#DROP TABLE fhmotitle_result IF EXISTS;
CREATE TABLE fhmotitle_result AS
SELECT title, pname
FROM (
	SELECT 
		CASE WHEN LENGTH(pname1) <= LENGTH(pname2) THEN pname1
		ELSE pname2
		END pname,
		a.*
	FROM (
		SELECT t.*, 
			SUBSTRING_INDEX(title, '-', -1) pname1, SUBSTRING_INDEX(title, ' ', -1) pname2
		FROM fhmotitle t
	) a
) b WHERE LENGTH(pname)<=20;

#再根据key反查没有归类的mo，将查到的结果补充给result
Python script

#最后从result中生成JSON
SELECT #m.*, 
	CONCAT('{''title'':''', title, ''', ''status'':''INIT'', ''keyname'':''', m.pname, '''},') json#, status, capture_date
FROM fhmotitle_result m 
ORDER BY capture_date DESC;

CREATE TABLE fhmotitle_key AS 
SELECT pname, COUNT(1) cnt
FROM (
	SELECT 
		CASE WHEN LENGTH(pname1) <= LENGTH(pname2) THEN pname1
		ELSE pname2
		END pname,
		a.*
	FROM (
		SELECT t.*, 
			SUBSTRING_INDEX(title, '-', -1) pname1, SUBSTRING_INDEX(title, ' ', -1) pname2
		FROM fhmotitle t
	) a
) b WHERE LENGTH(pname)<=20 #and pname regex'[0-9]'
GROUP BY pname
ORDER BY 2 DESC;
ORDER BY LENGTH(pname) DESC;

CREATE TABLE fhmokey(
	create_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	keyname VARCHAR(100),
	PRIMARY KEY(keyname)
);

SELECT * FROM fhmokey;
SELECT * FROM fhmotitle;
SELECT * FROM fhmotitle_result;
SELECT * FROM fhmotitle_key;
