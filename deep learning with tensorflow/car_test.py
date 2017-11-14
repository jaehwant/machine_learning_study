import numpy as np
import tensorflow as tf

data = np.loadtxt('cars.csv', unpack= True, delimiter=',')

data[0] # 마력
data[1] # 배기량

print(data[0])
print(data[1])

w = tf.Variable([1.0],tf.float32)
b = tf.Variable([1.0], tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

H = x * w + b# 가설 설정

cost = tf.reduce_sum(tf.square(H - y))

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(cost)

x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1000):
    sess.run(train,{x:data[0],y:data[1]})

curr_W, curr_b, curr_cost = sess.run([w,b,cost],{x:data[0],y:data[1]})

print("W: %s, b: %s, loss: %s" % (curr_W, curr_b, curr_cost))

# cost/loss function

#linear regression model 생성
# 마력 :10 , 20, 30, 40
# 배기량 : ?, ?, ?, ?,

# 메일로 전달
# 배기량 출력결과, 코드  egohimdrura@daum.net :여기로 전달.