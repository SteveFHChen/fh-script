



1 queries executed, 1 success, 0 errors, 0 warnings
查询：ALTER TABLE sec_stock_history DROP PRIMARY KEY
共 13568608 行受到影响
执行耗时   : 4 min 59 sec
传送时间   : 1.014 sec
总耗时      : 5 min

1 queries executed, 1 success, 0 errors, 0 warnings
查询：ALTER TABLE sec_stock_history ADD PRIMARY KEY(price_date, stock_code)
共 0 行受到影响
执行耗时   : 6 min 33 sec
传送时间   : 1.009 sec
总耗时      : 6 min 34 sec

查询时where条件中有索引字段过滤会快很多；
多字段构成的组合索引，索引中字段的顺序很重要，决定了查询的性能。
