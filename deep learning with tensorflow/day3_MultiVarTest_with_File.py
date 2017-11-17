import tensorflow as tf
import numpy as np

tf.set_random_seed(777)
xy = np.loadtxt('score.csv', delimiter=',', dtype=np.float32)
xdata = xy[:, 0:-1]
ydata = xy[:, [-1]]
print(xy.shape, xy, len(xy))
print(xdata.shape, xdata, len(xdata))
print(ydata.shape, ydata, len(ydata))


# feed_dict로 값을 받을 placeholder 생성
x = tf.placeholder(tf.float32, shape=[None,3]) # Shape 에서 앞 인자는 instance의 갯수,
y = tf.placeholder(tf.float32,shape=[None,1])


w = tf.Variable(tf.random_normal([3,1]), name="Weight")
b = tf.Variable(tf.random_normal([1]), name="Bias")

# 가설 함수 정의
hypothsis = tf.matmul(x,w)+b

# cost/loss 함수 정의
cost = tf.reduce_mean(tf.square(hypothsis - y))

# learning rate 을 반영해서 minimize 한다.
learning_rate = 1e-5
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20001):
    cost_val, hf_val, _ = sess.run([cost, hypothsis, train], feed_dict={x: xdata, y:ydata})
    if step % 1000 == 0:
        print(step, "Cost = " + str(cost_val) + " Prediction : \n" + str(hf_val))


print("Test")
print(sess.run(hypothsis,feed_dict={x:[[10,20,30]]}))


