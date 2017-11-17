import tensorflow as tf
import numpy as np

tf.set_random_seed(777)
# shuffle false 이면 파일 내의 순서가 동일하다.
filename_queue = tf.train.string_input_producer(['score.csv'], shuffle=False, name="filename_queue")

reader = tf.TextLineReader()
key,value = reader.read(filename_queue) # key는?, value 에 실제 데이터가 들어간다.
record_defaults = [[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value,record_defaults=record_defaults)
#score 데이터가 csv포멧으로 디코딩 됨.

train_x_batch,train_y_batch = tf.train.batch([xy[0:-1],xy[-1:]], batch_size =10) #0에서 -1까지
#x입력데이터는 train_x_batch로, y 입력데이터는 train_y_batch로 들어가게 된다.

x = tf.placeholder(tf.float32,shape=[None,3]) #x값의 갯수
y = tf.placeholder(tf.float32,shape=[None,1]) #y값의 갯수

w = tf.Variable(tf.random_normal([3,1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name="bias")

#가설 정의
hf = tf.matmul(x,w) + b

#비용함수 정의
cost = tf.reduce_mean(tf.square(hf-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

#세션 생성, 실행
sess =tf.Session()
sess.run(tf.global_variables_initializer())

#모델 생성할때
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(201):
    train_x_batch2,train_y_batch2 = sess.run([train_x_batch,train_y_batch])
    costv, hfv, _ = sess.run([cost,hf,train],feed_dict={x:train_x_batch2,y:train_y_batch2})

    if step % 10 == 0:
            print(step, "cost: " ,costv , "\nprediction\n", hfv)

#모델 종료할때 호출해야 함.
coord.request_stop()
coord.join(threads)
