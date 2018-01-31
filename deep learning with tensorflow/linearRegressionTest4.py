import tensorflow as tf
tf.set_random_seed(777)

w=tf.Variable([.3], tf.float32)
b=tf.Variable([.2], tf.float32)


x=tf.placeholder(tf.float32, shape=[None])
y=tf.placeholder(tf.float32, shape=[None])

hf=x*w+b
loss=tf.reduce_sum(tf.square(hf-y))
#loss=tf.reduce_mean(tf.square(hf-y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(loss)

xdata=[1,3,5,7,9]
ydata=[0,-2,-4,-8,-10]

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1000):
    sess.run(train, {x:xdata, y:ydata})

cw, cb, cl=sess.run([w,b,loss], {x:xdata, y:ydata})
print("w:%s b:%s loss:%s" % (cw, cb, cl))













