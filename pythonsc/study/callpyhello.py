
#import from other folder
import sys
sys.path.append('sklearn/')

#import funchello
import funchello as fh
from funchello import add

from classhello import Hello

print("Test one python file calls another:")

print("3+4=", fh.add(3,4))

print("5+6=", add(5,6))

print("Test class:")
h1=Hello(7, 8)
h1.add()


