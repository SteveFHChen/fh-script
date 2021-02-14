from pymysql import *
from properties import *

conn = connect(host=dbinfo["host"], port=dbinfo["port"], database=dbinfo["database"],
               user=dbinfo["user"],
               password=dbinfo["password"], charset=dbinfo["charset"])
               
cs1 = conn.cursor()
