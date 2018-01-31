#텐서플로우 그래프 만들기
import tensorflow as tf #default graph 가 생성, tf.get_default graph()
x=tf.linspace(-1.0,1.0,10)#구간 사이의 값을 생성
print(x)
g=tf.get_default_graph()
print(g)
print([op.name for op in g.get_operations()])
sess=tf.Session()
print(sess.run(x))
sess.close()

a=tf.placeholder("float")
b=tf.placeholder("float")
y=tf.multiply(a,b)
z=tf.add(y,y)
#elms=tf.Variable([1.0, 2.0, 2.0, 2.0])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(y, feed_dict={a:2,b:3}))
    print(sess.run(z, feed_dict={a:4,b:3,y:5}))


input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)
output=tf.multiply(input1, input2)
with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:5, input2:3} ))
#with구문으로 세션객체 생성
#input1, 2에 각각 5와 3을 전달하여 출력




