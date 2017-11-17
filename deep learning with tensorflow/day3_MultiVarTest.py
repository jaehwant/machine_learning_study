import tensorflow as tf

tf.set_random_seed(777)

x1data = [70, 90, 90, 95, 75]
x2data = [80, 87, 90, 97, 70]
x3data = [74, 94, 92, 99, 67]

ydata = [150, 186, 180, 195, 140]

# feed_dict로 값을 받을 placeholder 생성
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설 함수 정의
hypothsis = x1 * w1 + x2 * w2 + x3 * w3 + b

# cost/loss 함수 정의
cost = tf.reduce_mean(tf.square(hypothsis - y))

# learning rate 을 반영해서 minimize 한다.
learning_rate = 1e-5
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20001):
    cost_val, hf_val, _ = sess.run([cost, hypothsis, train], feed_dict={x1: x1data, x2: x2data, x3: x3data, y: ydata})
    if step % 100 == 0:
        print(step, "Cost = " + str(cost_val) + " Prediction : \n" + str(hf_val))

print("W1 : " + str(w1) + ", W2 : " + str(w2) + ", W3 : " + str(w3) + ", Bias : " + str(b))
