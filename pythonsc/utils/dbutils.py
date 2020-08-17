from pymysql import *


conn = connect(host='localhost', port=3306, database='test',
               user='root',
               password='root123', charset='utf8')

