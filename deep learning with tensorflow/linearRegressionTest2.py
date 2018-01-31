import tensorflow as tf
tf.set_random_seed(777)
x=[1,2,3]
y=[1,2,3]
w=tf.Variable(tf.random_normal([1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')

hf=x*w+b
loss=tf.reduce_mean(tf.square(hf-y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-10)
train=optimizer.minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2000):
    sess.run(train)
    if step % 100 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))












