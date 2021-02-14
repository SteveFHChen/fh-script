import pymysql
from DBUtils.PooledDB import PooledDB
from properties import *

dbPool = PooledDB(pymysql,50,
    host=dbinfo["host"],
    user=dbinfo["user"],passwd=dbinfo["password"],
    db=dbinfo["database"],port=dbinfo["port"]) 
    #50为连接池里的最少连接数

