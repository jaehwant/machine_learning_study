# import math
# print(math.sqrt(2.0))

class Man:
    def __init__(self,name): #초기화 함수, 생성자
        self.name = name
        print("Initialized")

    def hello(self):
        print("hello " + self.name)

    def goodbye(self):
        print("Good-bye"+self.name)

m = Man("John")
m.hello()
m.goodbye()
print(m.name)