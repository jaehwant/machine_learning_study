import tensorflow as tf
hello = tf.constant('hello') #constant 변수 생성
sess = tf.Session() # Session을 생성한다., 그래프를 실행시킬 수 있는 Session 객체를 만든다.
print(sess.run(hello)) # run 하는 시점에 그래프(session) 실행이 이루어 진다.

print(dir(tf))