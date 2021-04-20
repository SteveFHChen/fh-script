import time
import sys

#Way 1: multi-thread
import threading
#Way 2: multi-process
import multiprocessing

#import DBUtils.PooledDB as PooledDB
from DBUtils.PooledDB import PooledDB

#Way 1: MySQLdb
#import MySQLdb
#Way 2: pymysql
from pymysql import *
import pymysql

connargs = {
    "host": "localhost", 
    "port": 3306,
    "database": "test",
    "user": "root",
    "password": "root123", 
    "charset": "utf8"
    }
dbinfo = connargs

#processPool = multiprocessing.Pool(processes=5)#Cannot define at here
#dbPool = DBUtils.PooledDB.PooledDB(pymysql, **connargs)
#dbPool = PooledDB(pymysql, **connargs)
"""
dbPool = PooledDB(pymysql,50,
    host=dbinfo["host"],
    user=dbinfo["user"],passwd=dbinfo["password"],
    db=dbinfo["database"],port=dbinfo["port"]) 
    #这种写法不对，会不断报错。
"""
dbPool = PooledDB(pymysql, **connargs)

def test(conn, i):
    try:
        cursor = conn.cursor()
        count = cursor.execute("select * from fhmokey")
        rows = cursor.fetchall()
        for r in rows: pass
            #print(r)
        time.sleep(1)
    finally:
        conn.close()
        print(f"Connection {i} - {conn} closed.")

def testloop(i):
    print ("testloop")
    for j in range(1):
        #conn = MySQLdb.connect(**connargs)
        conn = connect(**connargs)
        print(f"Thread/Process {i} got connection {conn} to test...")
        test(conn, i)

def testpool(i):
    print ("testpool")
    #dbPool = DBUtils.PooledDB.PooledDB(MySQLdb, **connargs)
    
    for j in range(1):
        conn = dbPool.connection()
        print(f"Thread/Process {i} got connection {conn} to test...")
        test(conn, i)

def main(pp):
    t = testloop if len(sys.argv) == 1 else testpool
    
    print('Test using multi-thread...')
    for i in range(10):
        print(f"start thread {i}...")
        threading.Thread(target = t, args=(i, )).start()

    
    print('Test using multi-process...')
    for i in range(10, 20):
        pp.apply_async(testpool, args=(i, ))
        
if __name__ == "__main__":
    processPool = multiprocessing.Pool(processes=5)
    #Process pool define should be in __main__, otherwise exception will be throat.
    
    main(processPool)
    
    processPool.close()
    processPool.join()
    print("Process pool closed.")
    
    dbPool.close()
    print("DB pool closed.")
