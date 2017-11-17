import tensorflow as tf

tf.set_random_seed(777)

x1data = [[70, 90, 90],
          [95, 75 ,80],
          [87, 90, 97],
          [70,74, 94],
          [92, 99, 67]]

ydata = [[150],
         [186],
         [180],
         [195],
         [140]]

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

for step in range(2001):
    cost_val, hf_val, _ = sess.run([cost, hypothsis, train], feed_dict={x: x1data, y:ydata})
    if step % 100 == 0:
        print(step, "Cost = " + str(cost_val) + " Prediction : \n" + str(hf_val))



x1verify = [70, 90, 90, 95, 75]
x2verify = [80, 87, 90, 97, 70]
x3verify = [74, 94, 92, 99, 67]

yverify = [150, 186, 180, 195, 140]