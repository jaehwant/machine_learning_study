import tensorflow as tf
tf.set_random_seed(777)

xdata = [[0,0],
         [0,1],
         [1,0],
         [1,1]]
ydata = [[0],
         [1],
         [1],
         [0]]

x = tf.placeholder(tf.float32,shape=[None,2])
y = tf.placeholder(tf.float32,shape=[None,1])

# w = tf.Variable(tf.random_normal([2,1],name="weight"))
# b = tf.Variable(tf.random_normal([1],name="bias"))

# 1계층
w1 = tf.Variable(tf.random_normal([2,10],name="weight1")) #앞의 2의 의미는 x의 갯수, 뒤의 2는 출력의 갯수
b1 = tf.Variable(tf.random_normal([10],name="bias1"))
layer1 = tf.nn.relu(tf.matmul(x,w1)+b1)

#Deep NN
w2 = tf.Variable(tf.random_normal([10,10],name="weight2")) #앞의 2의 의미는 x의 갯수, 뒤의 2는 출력의 갯수
b2 = tf.Variable(tf.random_normal([10],name="bias2"))
layer2 = tf.nn.relu(tf.matmul(layer1,w2)+b2)

w3 = tf.Variable(tf.random_normal([10,10],name="weight3")) #앞의 2의 의미는 x의 갯수, 뒤의 2는 출력의 갯수
b3 = tf.Variable(tf.random_normal([10],name="bias3"))
layer3 = tf.nn.relu(tf.matmul(layer2,w3)+b3)

w4 = tf.Variable(tf.random_normal([10,1],name="weight4")) #앞의 2의 의미는 x의 갯수, 뒤의 2는 출력의 갯수
b4 = tf.Variable(tf.random_normal([1],name="bias4"))

hf = tf.sigmoid(tf.matmul(layer3,w4)+b4)
cost = -tf.reduce_mean(y*tf.log(hf) + (1-y)*tf.log(1-hf) )

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hf >0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001) :
        sess.run(train,feed_dict={x:xdata,y:ydata})
        if step % 100 == 0:
            print(step,sess.run(cost,feed_dict={x:xdata,y:ydata}))
    h,c,a = sess.run([hf,predicted,accuracy],feed_dict={x:xdata,y:ydata})
    print("\n가설 : ",h,"\ncorrect : ",c,"\n정확도 : ",a)