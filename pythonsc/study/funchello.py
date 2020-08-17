
'''
1. 自定义函数
2. 函数形参的定义方式：
　　1.位置形参
　　2.型号的元组形参
　　3.命名关键字形参
　　4.双星号字典形参
'''

import sys

def add(x, y):
    return x+y
    
def addArray(x, y):
    return [x, y, x+y]

def addJson(p):
    return {"x": p["x"], "y":p["y"], "sum": p["x"]+p["y"]}

s=add(2,3)
print("s:",s)

s1=addArray(4,5)
print("s1:",s1)

p1={"x":6, "y":7}
s2=addJson(p1)
print("s2:",s2)

a=int(input("请输入第1个整数："))
b=int(input("请输入第2个整数："))
print("a+b=", add(a, b))

