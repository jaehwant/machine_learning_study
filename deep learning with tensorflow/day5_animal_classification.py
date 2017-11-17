import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt('animal_data.csv', delimiter=',', dtype=np.float32)

xdata = xy[:,0:-1]
ydata = xy[:,[-1]]

print(xdata.shape, ydata.shape)

nb_class = 7
x  = tf.placeholder(tf.float32,[None,16])
y = tf.placeholder(tf.int32,[None,1])

# 0 ~ 6까지 숫자를 ne-hot 으로 변경 해야 함. ex) 2 -> [0010000]
#tf.one_hot(원핫인코딩 대상, )
y_one_hot = tf.one_hot(y,nb_class) #차원이 1 증가\
print("one hot : ",y_one_hot)
#예를 들어, [[0], [3]]이면 ?,1의 shape 를 갖게 되는 것이며, 이것을 one-hot 인코딩 하면
# [[[1000000]][[0001000]]]이 됩니다. 즉 rank가 2인 상태에서 one-hot 인코딩 하면 rank 가 3이 된다. (rank는 차원)
#우리가 원하는 shape은 (?,7)이다. 이때 reshape를 통해 shape를 변경할 수 있다.
#tf.reshape(y_one_hot,[-1][7] 하면 shape은 [?,7]이 됩니다.
#즉, [[1000000],[0001000]]이 최종적으로 변환된 결과가 됩니다.

y_one_hot = tf.reshape(y_one_hot,[-1,7])
print("reshape : ", y_one_hot)

w= tf.Variable(tf.random_normal([16,nb_class]),name="weight")
b= tf.Variable(tf.random_normal([nb_class]),name="bias")

logits = tf.matmul(x,w)+b
hf = tf.nn.softmax(logits)

costi = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
cost = tf.reduce_mean(costi)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hf,1) #proability -> 0~6 사이의 숫자로 변환
correct_prediction = tf.equal(prediction, tf.argmax(y_one_hot,1)) #같으면 True, 다르면 False , y label에서의 최대값
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#lunch Graph

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(optimizer,feed_dict={x:xdata,y:ydata})
        if step % 100 == 0:
            costv,acc = sess.run([cost,accuracy],feed_dict={x:xdata,y:ydata})
            print("step : {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step,costv,acc))
    pred = sess.run(prediction,feed_dict={x:xdata})

    for p, y in zip(pred, ydata.flatten()):
        print("[{}] prediction: {} True Y: {}".format(p ==int(y), p, int(y)))