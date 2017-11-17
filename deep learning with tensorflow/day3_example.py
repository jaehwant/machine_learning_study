'''
당뇨병 예측 모델 Logistic regression 구현
데이터는 : diabetes.csv 사용
'''

import tensorflow as tf
import numpy as np
xy = np.loadtxt('diabetes.csv',delimiter=',', dtype=np.float32)

x_train_data = xy[:700,0:-1]
y_train_data = xy[:700,[-1]]
x_verify_data = xy[701:,0:-1]
y_verify_data = xy[701:,[-1]]

print(x_train_data.shape)
print(y_train_data.shape)

print(x_verify_data.shape)
print(y_verify_data.shape)

x = tf.placeholder(tf.float32,shape=[None,8])
y = tf.placeholder(tf.float32,shape=[None,1])

w = tf.Variable(tf.random_normal([8,1],name="weight"))
b = tf.Variable(tf.random_normal([1]),name="bias")

hypothsis = tf.sigmoid(tf.matmul(x,w) + b) # sigmoid 를 적용한다. 가설의 결과값이 1 과 0 사이로 나온다.

cost = -tf.reduce_mean(y * tf.log(hypothsis) + (1-y) * tf.log(1-hypothsis))

predicted = tf.cast(hypothsis>0.5, tf.float32) # if hf>.5 == True 이면 1.0이 된다. else 이면 0이 됨.
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y),tf.float32)) # 함수의 결과가 True 아니면 False가 나온다.

learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000000):
        costv, _ = sess.run([cost,train],feed_dict={x:x_train_data,y:y_train_data})
        if step % 5000 == 0 :
            print("Cost : ", costv)

    h,c,a = sess.run([hypothsis,predicted,accuracy], feed_dict={x:x_train_data,y:y_train_data})
    print("accuracy:(train_data)", a)
    h, c, a = sess.run([hypothsis, predicted, accuracy], feed_dict={x: x_verify_data, y: y_verify_data})
    print("accuracy:(verify_data)", a)

    saver = tf.train.Saver()
    save_path = saver.save(sess, "./minist_softmax.ckpt")
    import os

    print(os.getcwd())
    print("Model saved in file: ", save_path)
# with tf.Session() as sess: # with문이 종료되면 자동으로 세션을 닫아준다.
#     sess.run(tf.global_variables_initializer())
#     for step in range(10001):
#         costv, _ = sess.run([cost,train],feed_dict={x:xdata,y:ydata})
#
#         if step % 200 == 0 :
#             print(step,costv)
#
#     h,c,a = sess.run([hf,predicted,accuracy], feed_dict={x:xdata,y:ydata})
#     print("가설 : ",h, "\nCorrect :", c, "\naccuracy:", a)