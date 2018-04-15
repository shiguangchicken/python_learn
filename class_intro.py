class Root:
    __total=0
    def __init__(self,v):
        self.__value=v
        Root.__total+=1

    def show(self):
        '''普通方法，一般以self作为第一个参数'''
        print('self.__value:',self.__value)
        print('Root.__total:',Root.__total)

    @classmethod         #修饰器，声明类方法
    def classShowTotal(cls):
        '''类方法，一般以cls作为第一个参数名称'''
        print(cls.__total)

    @staticmethod
    def staticShowTotal():
        '''修饰器，声明静态方法，静态方法可以没有参数'''
        print(Root.__total)

r=Root(3)
r.classShowTotal() #通过对象来调用方法
r.staticShowTotal() #通过静态方法来调用方法
rr=Root(5)
Root.classShowTotal()#通过类名调用方法
Root.staticShowTotal()#通过类名调用静态方法
Root.show()      #错误调用
Root.show(r)

'''以下是抽象方法'''
import abc
class Foo(metaclass=abc.ABCMeta):   #抽象类
    def f1(self):
        '''普通方法'''
        print(123)
    @abc.abstractclassmethod
    def f3(self):
        '''抽象方法'''
        raise Exception('you must reimlement this method')
class Bar(Foo):
    def f3(self):
        print(333) #必须重新实现基类中的抽象方法
b=Bar()
b.f3()

'''以下是属性'''

class Test:
    def __init__(self,value):
        self.__value=value      #私有数据成员

    @property                   #修饰器，定义属性
    def value(self):            #只读属性，无法修改和删除
        return self.__value

t=Test(3)
t.value
#t.value=5  错误
'''下面代码可设置属性可修改，不允许删除'''
class Test1:
    def __init__(self, value):
        self.__value = value  # 私有数据成员

    @property  # 修饰器，定义属性
    def __get(self):  # 只读属性，无法修改和删除
        return self.__value
    def __set(self,v):
        self.__value=v
    value=property(__get,__set())#可读可写属性，指定相应的读写方法
    '''value=property(__get,__set,__del)表示可读可写可删除'''
    def show(self):
        print(self.__value)
t = Test1(3)
t.value
t.value=5
t.show()
