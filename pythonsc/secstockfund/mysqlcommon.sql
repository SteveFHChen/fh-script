SELECT NOW();
SELECT UNIX_TIMESTAMP(NOW());
SELECT UNIX_TIMESTAMP(NOW())*1000;
SELECT UNIX_TIMESTAMP('2018-01-01')*1000;
SELECT FROM_UNIXTIME(1609257600000/1000);
#mysql中表字段为releaseTime　bigint(20)存的是精确到毫秒的java timestamp。必段除以1000，否则返回null。

SELECT * FROM parameters;
SET GLOBAL max_connections=500;
SHOW VARIABLES LIKE '%connect%';
SHOW PROCESSLIST;

SELECT DATE_SUB(CURDATE(), INTERVAL 5 DAY);
SELECT DATE_SUB(DATE('2021-02-03'), INTERVAL 5 DAY);

