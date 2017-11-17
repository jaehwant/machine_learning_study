import tensorflow as tf
import matplotlib.pyplot as plt
import random
tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

nb_class = 10

#mnist 이미지의 shape 28*28 = 784

x = tf.placeholder(tf.float32,shape=[None,784]) # 픽셀 수
y = tf.placeholder(tf.float32,shape=[None,nb_class]) # 출력 숫자 갯수

w = tf.Variable(tf.random_normal([784,nb_class]),name="Weight")
b = tf.Variable(tf.random_normal([nb_class]),name="Bias")

hf = tf.nn.softmax(tf.matmul(x,w)+b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hf),axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
is_correct = tf.equal(tf.argmax(hf,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

trainging_epochs = 20 #에포크 : 전체 데이터 셋을 20번 반복하여 트레이닝 한다.
batch_size = 100 # 한번에 올릴수는 없고, 한번에 몇개씩 메모리에 올려서 트레이닝 할 것인지, 이미지의 갯수가 됨.
                 # 한번에 메모리에 복사하여 트레이닝하고자 하는 대상 이미지의 수

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for epoch in range(trainging_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size) # 전체 데이터의 수 / 100, 전체데이터의 수가 5천이면, 50이겠지. 50번 batch한다

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) # 100개씩 데이터를 읽어 온다.
            c, _ = sess.run([cost,optimizer],feed_dict={x:batch_xs,y:batch_ys})
            avg_cost += c/total_batch
        print('Epoch : ', '%04d' % (epoch+1),
              'Cost : ', '{:.9f}'.format(avg_cost))


    print("learning finished")
    #동시에 여러개 실행할때는 sess.run으로 해야 함. 한개만 하고 싶을때는 .eval로 가능
    print("Accuracy:",accuracy.eval(session=sess, feed_dict={
        x:mnist.test.images, y:mnist.test.labels
    }))

    r = random.randint(0,mnist.test.num_examples-1) # 임의의 숫자 이미지의 위치
    print("Label : ",sess.run(tf.argmax(mnist.test.labels[r:r+1],1))) #임의의 숫자에 대한 실제 레이블을 가져옴.
    print("Prediction : ", sess.run(tf.argmax(hf,1), feed_dict={x:mnist.test.images[r:r+1]}))
    plt.imshow(mnist.test.images[r:r+1].reshape(28,28),
               cmap="Greys",
               interpolation="nearest")
    plt.show()
