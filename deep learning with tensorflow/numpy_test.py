import tensorflow as tf

a = tf.constant([5],tf.float32)
b = tf.constant([10],tf.float32)
c = tf.constant([2],tf.float32)
d = a*b+c

sess = tf.Session()
res = sess.run(d)
print(res)

# Tensor("Const:0", shape=(1,), dtype=float32) 출력결과 ,
# Tensor("Const_1:0", shape=(1,), dtype=float32)
# Tensor("Const_2:0", shape=(1,), dtype=float32)
# Tensor("add:0", shape=(1,), dtype=float32)


# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.arange(start=0, stop=6,step=0.1)
# y1 = np.sin(x)
# y2 = np.cos(x)
#
# plt.plot(x,y1,label="sin")
# plt.plot(x,y2,linestyle="--",label="cos")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("sin&Cos function")
# plt.legend()
# plt.show()


# 연습1.
#  import numpy  as np
# x  = np.array([1.0, 2.0, 3.0])
# y = np.array([2.0,4.0,6.0])
#
# print(x+y) # element-wise 엘리먼트 단위의 연산.
# print(x-y)
# print(x/2.0)
#
# A = np.array([[1,2],[3,4]]) # rank가 2, shape = 2
# print(A)
# print(A.shape) #어떤모습의 행렬인지 확인하는 방법.
#
# B = np.array([[3,0],[0,6]])
# print(A+B)
# print(A*B)
# x = np.array([[1.0,2.0], [3.0,4.0],[5.0,6.0]])
# print(x)
# print(x[0])
#
# print("==============")
# for row in x:
#     print(row)
# x =x.flatten() # 2차원 배열을 1차원행렬로 변환시킨다.ㄴ
# print(x)
# print(x[np.array([0,3,5])]) # array의 0번째, 3번째, 5번째 데이터를 출력한다.
# print(x>3) # 각각의 요소마다 비교 연산이 된다.
#
# print(x[x>3]) # true에 해당되는 것만 실행된다.