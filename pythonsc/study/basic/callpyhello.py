
#import from other folder
import sys
sys.path.append('myclass/')

#import funchello
import funchello as fh
from funchello import add

from classhello import Hello
from classhellosub import *

print("Test one python file calls another:")

print("3+4=", fh.add(3,4))

print("5+6=", add(5,6))

print("Test class:")
print("(1) Test simple class:")
h1=Hello(7, 8)
h1.add()

print("(2) Test sub class:")
hs1=HelloSub(3,5)
hs1.add()
hs1.sub()



