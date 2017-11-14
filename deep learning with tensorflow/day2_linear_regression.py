# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
#
# num_points = 1000
# vectors_set = []
# for i in range(num_points):
#     x1 =  np.random.normal(0.0,0.5) # random 함수인데 정규분포를 따르는 랜덤한 수를 만든다.
#     y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0,0.3)
#     vectors_set.append([x1,y1])
#
# # x data와 y_data를 만들어놓은 상태
# x_data = [v[0] for v in vectors_set]
# y_data = [v[1] for v in vectors_set]
#
# plt.plot(x_data,y_data,'ro')
# plt.legend()
# plt.show()
#
# w = tf.Variable(tf.random_uniform([1],-1.0,1.0))# -1 에서 1까지의 랜덤한 값을 하나 뽑는다. random normal은 정규분포로, uniform은 균등하게 난수를 생성한다.
# b = tf.Variable(tf.zeros([1])) # 0으로 초기화 , ones 는 1로 초기화.
# y = w * x_data + b
#
# cost = tf.reduce_mean(tf.square(y - y_data))
#
# # cost 함수까지 정의함.
# optimizer = tf.train.GradientDsaescentOptimizer(0.5)
# train = optimizer.minimize(cost)
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
#
# for step in range(8):
#     sess.run(train)
#     print(step, sess.run(w),sess.run(b))
#     print(step,sess.run(cost))
#     #시각화
#     plt.plot(x_data,y_data,'ro')
#     plt.plot(x_data,sess.run(w)*x_data+sess.run(b))
#     plt.xlabel()
#     plt.xllm(-2,2)
#     plt.ylim(0.1,0.6)

# import tensorflow as tf
# tf.set_random_seed(777)
#
# x = [1,2,3]
# y = [1,2,3]
# w = tf.Variable(tf.random_normal([1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')
#
# hf =  x * w + b
# loss = tf.reduce_mean(tf.square(hf-y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for step in range(2000) :
#     sess.run(train)
#     if step % 100 == 0:
#         print(step, sess.run(loss),sess.run(w),sess.run(b))

# import tensorflow as tf
# tf.set_random_seed(777)
#
#
# w = tf.Variable(tf.random_normal([1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')
#
# x = tf.placeholder(tf.float32, shape=[None])
# y = tf.placeholder(tf.float32, shape=[None])
#
# hf =  x * w + b
# loss = tf.reduce_mean(tf.square(hf-y))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for step in range(2000) :
#     lossv, wv, bv, _  = sess.run([loss,w,b,train],feed_dict={x:[1,2,3],y:[1,2,3]})
#     if step % 100 == 0:
#          print(step, lossv, wv, bv)

# import tensorflow as tf
# tf.set_random_seed(777)
#
#
# w = tf.Variable([0.3],tf.float32)
# b = tf.Variable([10.0], tf.float32)
#
# x = tf.placeholder(tf.float32, shape=[None])
# y = tf.placeholder(tf.float32, shape=[None])
#
# hf =  x * w + b
# loss = tf.reduce_sum(tf.square(hf-y))
# # loss = tf.reduce_mean(tf.square(hf-y))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
# train = optimizer.minimize(loss)
#
# xdata = [1,3,5,7,9]
# ydata= [0,-2,-4,-8, -10]
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for step in range(5000) :
#     sess.run(train,{x:xdata, y:ydata})
# cw, cb, cl = sess.run([w,b,loss],{x:xdata,y:ydata})
#
# print("w:%s, b:%s, loss:%s" % (cw, cb, cl))

# import tensorflow as tf
# # placeholder를 적용해서 특정 단을 출력하는 프로그램을 작성해보세요.?
# # 3*1 = 3
# # ...
# # 3*9 = 27
#
# def dan(num):
#     right = tf.placeholder(dtype=tf.int32)
#     calc =tf.multiply(num,right)
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#
#     for i in range(1,10):
#         res = sess.run(calc,feed_dict={right:i})
#         print('{}x{}={:2}'.format(num,i,res))
#     sess.close()
#
# dan(9)


#cars.cvs를 가지고 