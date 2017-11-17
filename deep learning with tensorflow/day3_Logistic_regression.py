import tensorflow as tf
tf.set_random_seed(777)

xdata = [[1,2,],
         [2,3],
         [3,1],
         [4,3],
         [5,3],
         [6,2]]
ydata = [[0],
         [0],
         [0],
         [1],
         [1],
         [1]]

x = tf.placeholder(tf.float32,shape=[None,2])
y = tf.placeholder(tf.float32,shape=[None,1])

w = tf.Variable(tf.random_normal([2,1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name="bias")

hf = tf.sigmoid(tf.matmul(x,w) + b) # 0 에서 1사이의 값으로 나온다.

cost  = - tf.reduce_mean(y*tf.log(hf) + (1-y)*tf.log(1-hf))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hf>0.5, tf.float32) # if hf>.5 == True 이면 1.0이 된다. else 이면 0이 됨.
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y),tf.float32)) # 함수의 결과가 True 아니면 False가 나온다.

with tf.Session() as sess: # with문이 종료되면 자동으로 세션을 닫아준다.
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        costv, _ = sess.run([cost,train],feed_dict={x:xdata,y:ydata})

        if step % 200 == 0 :
            print(step,costv)

    h,c,a = sess.run([hf,predicted,accuracy], feed_dict={x:xdata,y:ydata})
    print("가설 : ",h, "\nCorrect :", c, "\naccuracy:", a)