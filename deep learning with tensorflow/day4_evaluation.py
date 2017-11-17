import tensorflow as tf
tf.set_random_seed(777)

xdata = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
ydata = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]


xtest = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
ytest = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]


x = tf.placeholder(tf.float32,shape=[None,3])
y = tf.placeholder(tf.float32,shape=[None,3])

w = tf.Variable(tf.random_normal([3,3]), name="weight")
b = tf.Variable(tf.random_normal([3]),name="bias")

hf = tf.nn.softmax((tf.matmul(x * w) + b))
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hf), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.argmax(hf,1) # argmax를 했더니 0으로 나온다. 그럼 0으로 예측했다?
is_correct = tf.equal(prediction, tf.argmax(hf,1))

accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32)) #몇개가 맞는지, Accuracy가 된다.

with tf.Session() as sess: # with에 색션개채를 만들면, program이 종료시 resource가 자동으로 close된다.
    sess.run(tf.global_variables_initializer())

    for step in range(2001) :
        costv,wv, _ = sess.run([cost,w,optimizer],feed_dict={x:xdata,y:ydata})

        print(step,costv,wv)
        


