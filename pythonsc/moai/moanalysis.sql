/* Data analysis */
#Check the number movies of each key
SELECT * 
FROM fhmokey k
LEFT JOIN (
	SELECT keyname, COUNT(1)
	FROM fhmoinfo
	GROUP BY keyname
	ORDER BY 2 DESC
) m ON k.keyname = m.keyname
ORDER BY 4 DESC;

UPDATE fhmoinfo
SET STATUS='FINISHED', capture_date=CURRENT_TIMESTAMP
#select * from fhmoinfo
WHERE title IN (
''
);

/* Generate JSON string for frontend */
SELECT CONCAT('{''key'': ''', k.pname, ''', ''total'': ', cnt, '},') json, cnt
#SELECT * 
FROM fhmotitle_key k
ORDER BY cnt DESC;

SELECT CONCAT('{''key'': ''', k.keyname, ''', ''total'': ', total, '},') json#, total
#SELECT * 
FROM fhmokey k
LEFT JOIN (
	SELECT keyname, COUNT(1) total
	FROM fhmoinfo
	GROUP BY keyname
	ORDER BY 2 DESC
) m ON k.keyname = m.keyname
ORDER BY create_date DESC;


SELECT #m.*, 
	CONCAT('{''title'':''', title, ''', ''status'':''', STATUS,''', ''keyname'':''', keyname, '''},') json#, status, capture_date
FROM fhmoinfo m
ORDER BY capture_date DESC;



