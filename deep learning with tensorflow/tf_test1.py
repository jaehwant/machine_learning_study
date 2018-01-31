import tensorflow as tf
a=tf.constant(1)
print(a)
with tf.Session() as sess:
    print(a.eval())

b=tf.constant(5,name="input_b")
c=tf.constant(3,name="input_c")
d=tf.multiply(b,c, name="mul_d")
e=tf.add(b,c, name="add_e")
f=tf.add(d,e,name="add_f")
sess=tf.Session()
print(sess.run(d))

g=tf.constant([10], dtype=tf.float32)
h=tf.constant([2], dtype=tf.float32)
i=g+h
print(sess.run(i))

input_data=[1,2,3,4,5]
x=tf.placeholder(dtype=tf.float32)
w=tf.Variable([3], dtype=tf.float32)
y=w*x
init=tf.global_variables_initializer()
sess.run(init)

result=sess.run(y, feed_dict={x:input_data})
print(result)










