# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:59:51 2021

@author: steve
"""
import time

class LogWriter:
    def __init__(self, logFile):
        self.logFile = logFile
        self.fLog = open(logFile, 'a+')
        
    def writeLog(self, msg):
        msgx = f'\n[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] {msg}'
        print(msgx)
        self.fLog.writelines([msgx])
        self.fLog.flush()
        
    def close(self):
        self.fLog.flush()
        self.fLog.close()
        
if __name__ == "__main__":
    logFile='C:/fh/testenv1/sec/stock.log'
    logger = LogWriter(logFile)
    logger.writeLog("Testing write log class")
    logger.close()
