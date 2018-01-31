import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)
x=[1,3,5,7,9]
y=[2,4,6,8,10]
w=tf.placeholder(tf.float32)
hfunc=x*w
loss=tf.reduce_mean(tf.square(hfunc-y))
sess=tf.Session()

wlist=[]
losslist=[]

for i in range(-50, 50):   #-5~5 범위 0.1씩 증가
        cw=i*0.1
        closs=sess.run(loss, feed_dict={w:cw})
        wlist.append(cw)
        losslist.append(closs)

plt.plot(wlist, losslist)
plt.show()











