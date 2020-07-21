import os#os 库是Python标准库，包含几百个函数，常用的有路径操作、进程管理、环境参数等。
import logging
LOG_FORMAT='%(asctime)s - %(levelname)s - %(message)s'
DATA_FORMAT='%Y-%m-%d %H:%M:%S'
logDir='./auto_pip.log'
logging.basicConfig(filename=logDir,level=logging.DEBUG,format=LOG_FORMAT,datefmt=DATA_FORMAT)
mirror_url=r'https://pypi.douban.com/simple/'
#1-读取配置文件  XXX.ini
fo=open('./pythonLibConfig.txt')#打开文件对象,./当前路径
lines=fo.read().splitlines()#读取返回是列表，没有换行符


#2-执行安装库的指令
backInfo=os.popen("pip list")
info=backInfo.read()
print(info)
for line in lines:
    print(line)
    if line in info:
        print('该库已经成功安装！')
        logging.warning(f'---------------->{line}installed!')
    else:
    #没有安装的库
        pipObject=os.popen('pip'+ ' '+'install'+' '+line+' '+'-i'+' '+mirror_url)
        pipRes=pipObject.read()
        if 'Successfully' in pipRes:
            print('安装成功---------->')
            logging.info(f'---------------->{line}installed pass!')
        else:
            print('安装失败---------->')
            logging.error(f'--------------->{line}installed failed!')
#3-验证安装是否成功
#4-打印日志-----------logging(内置模块)