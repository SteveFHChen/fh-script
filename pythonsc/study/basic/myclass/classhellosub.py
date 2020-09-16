
from classhello import *

class HelloSub(Hello):
    def sub(self):
        print("x-y=", self.x-self.y)

'''
Usage sample:
>>> hs=HelloSub(5,3)
>>> hs.add()
x+y= 8
>>> hs.sub()
x-y= 2
'''

#Test case

hs = HelloSub(5,3)
hs.add()
hs.sub()
