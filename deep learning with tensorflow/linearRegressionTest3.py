import tensorflow as tf
tf.set_random_seed(777)

w=tf.Variable(tf.random_normal([1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')

x=tf.placeholder(tf.float32, shape=[None])
y=tf.placeholder(tf.float32, shape=[None])

hf=x*w+b
loss=tf.reduce_mean(tf.square(hf-y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2000):
    lossv, wv, bv, _=sess.run([loss, w, b, train], feed_dict={x:[1,2,3], y:[1,2,3]})
    if step % 100 == 0:
        print(step, lossv, wv, bv)












