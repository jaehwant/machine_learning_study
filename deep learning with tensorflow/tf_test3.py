import tensorflow as tf
state=tf.Variable(0, name="counter")
one=tf.constant(1)
new_value=tf.add(state, one)
update=tf.assign(state,new_value)#변수 업데이트tf.assign(업데이트대상변수, 업데이트값)
init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))




# input_data=[[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [3.0, 2.0, 1.0]] #3*3 행렬
#
# x=tf.placeholder(dtype=tf.float32, shape=[None,3])  #1행 3열
# w=tf.constant([[2.0], [2.0], [2.0]]) #3행 1열
# y=tf.matmul(x,w)
# sess=tf.Session()
# init=tf.global_variables_initializer()
# sess.run(init)
# result=sess.run(y, feed_dict={x:input_data})
# print(result)

# import tensorflow as tf
# x=tf.Variable([[1.0, 2.0, 3.0]])  #1행 3열
# w=tf.constant([[2.0], [2.0], [2.0]]) #3행 1열
# y=tf.matmul(x,w)
# print(x.get_shape())
# sess=tf.Session()
# init=tf.global_variables_initializer()
# sess.run(init)
# result=sess.run(y)
# print(result)


# import tensorflow as tf
# x=tf.constant([[1.0, 2.0, 3.0]])  #1행 3열
# w=tf.constant([[2.0], [2.0], [2.0]]) #3행 1열
# y=tf.matmul(x,w)
# print(x.get_shape())
# sess=tf.Session()
# init=tf.global_variables_initializer()
# sess.run(init)
# result=sess.run(y)
# print(result)
