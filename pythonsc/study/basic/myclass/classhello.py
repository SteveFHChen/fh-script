
class Hello:
    def __init__(self, x, y):
        self.x=x
        self.y=y
    def add(self):
        print("x+y=", self.x+self.y)

'''
>>> Hello
<class '__main__.Hello'>
>>> dir(Hello)
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'add']
>>> help(Hello.add)
Help on function add in module __main__:

add(self)

>>> help(Hello)
Help on class Hello in module __main__:

class Hello(builtins.object)
 |  Hello(x, y)
 |
 |  Methods defined here:
 |
 |  __init__(self, x, y)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |
 |  add(self)
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |
 |  __dict__
 |      dictionary for instance variables (if defined)
 |
 |  __weakref__
 |      list of weak references to the object (if defined)
'''
