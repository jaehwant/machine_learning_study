import tensorflow as tf
#
# # tf.placeholder() #실행중에 데이터를 Feed(전달)해서
# #Tensor에 할당된 placeholder로 정의해도 Tensor 결정되고 실행시 데이터가 할당되어야 함.
# a = tf.constant(1)
# print(a)
#
# sess = tf.Session()
#
# # b = tf.constant(5,name="input_b")
# # c = tf.constant(3,name="input_c")
# # d = tf.constant(b,c, name="mul_d")
# # e = tf.add(b,c, name ="add_e")
# # f = tf.add(d,e, name ="add_f")
# # sess = tf.Session()
# # print(sess.run(d))
#
# g = tf.constant([10], dtype=tf.float32)
# h = tf.constant([2], dtype=tf.float32)
# i = g +h
# print(sess.run(i))
#
# input_data = [1,2,3,4,5]
# x = tf.placeholder(dtype=tf.float32)
# W = tf.Variable([3],dtype=tf.float32)
#
# y = W * x
#
# init = tf.global_variables_initializer()
#
# sess.run(init)
#
# result = sess.run(y, feed_dict={x:input_data})
# print(result)

# 텐서플로우 용어, 텐서 == 다차원 배열
# 내부 자료구조는 텐서로 표현한다.
# Session은 그래프를 실행시키는 실행 객체이다, run()로 실행
# op(오퍼레이션) : 동작에 대한 정의, 텐서를 입력 받아서 연산한 후 결과를 반환
# 그래프 : 연산을 표현한 것
# 노드 : 오퍼레이션의 정의에 포함되는 것
# 엣지 : 노드간의 연결하는 선

#텐서 플로우 그래프 만들기

# import tensorflow as tf # default graph 가 생성된다. tf.get_default_graph()
# x = tf.linspace(-1.0,1.0,10) #구간 ~1 부터 1까지 10개를 만들겠다. 균등하게 나눠서 값을 출력한다.
# print(x)
# g = tf.get_default_graph() # import 시점에 기본적으로 만들어진 그래프가 리턴된다.
# print(g)
# print([op.name for op in g.get_operations()])
# sess = tf.Session()
# print(sess.run(x))
# sess.close()

# a = tf.placeholder("float")
# b = tf.placeholder("float")
# y = tf.multiply(a,b)
# z = tf.add(y,y)
# # elms = tf.Variable([1.0,2.0,2.0,2.0])
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(y, feed_dict={a:2,b:3}))
#     print(sess.run(z, feed_dict={a:4,b:3,y:5}))

# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
# output = tf.multiply(input1,input2)
# #with 구문으로 session 객체를 생성해서 input1, input2의 각각  5와 3을 전달하여 출력하기
#
# with tf.Session() as sess:
#     print(sess.run(output,feed_dict={input1:5,input2:3}))

#행렬표현예제

# x = tf.constant([[1.0,2.0,3.0]]) # 1행 3열
# w = tf.constant([[2.0],[2.0],[2.0]]) # 3행 1열
# y = tf.matmul(x,w)
#
# print(x.get_shape())
# print(w.get_shape())
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# result = sess.run(y)
# print(result)

# x = tf.Variable([[1.0,2.0,3.0]]) # 1행 3열
# w = tf.Variable([[2.0],[2.0],[2.0]]) # 3행 1열
# y = tf.matmul(x,w)
#
# print(x.get_shape())
# print(w.get_shape())
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# result = sess.run(y)
# print(result)

# input_data = [[[1.0,2.0,3.0], [2.0,2.0,3.0], [3.0,2.0,1.0]]] # 3행 3열, 데이터셋이 3개고, 각 데이터셋마다 3개의 feature가 있다.
# x = tf.placeholder(dtype=tf.float32, shape=[None,3]) # instance 3개, feature 3개 있다. 보편적으로 None을 많이준다. 왜냐면 Instance의 갯수는 가변적이기 때문에
# w =tf.constant([[2.0],[2.0],[2.0]])
# y = tf.matmul(x,w)
#
# print(x.get_shape())
# print(w.get_shape())
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# result = sess.run(y)
# print(result)

# input_data = [[[1.0,2.0,3.0], [2.0,2.0,3.0], [3.0,2.0,1.0]]] # 3행 3열, 데이터셋이 3개고, 각 데이터셋마다 3개의 feature가 있다.
# x = tf.placeholder(dtype=tf.float32, shape=[None,3]) # instance 3개, feature 3개 있다. 보편적으로 None을 많이준다. 왜냐면 Instance의 갯수는 가변적이기 때문에
# w =tf.constant([[2.0],[2.0],[2.0]])
# y = tf.matmul(x,w)
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# result = sess.run(y,feed_dict={x:input_data})
# print(result)

# state = tf.Variable(0, name="counter")
# one = tf.constant(1)
# new_value = tf.add(state,one)
# update = tf.assign(state,new_value) #assign을 통해서 변수를 업데이트를 할 수 있다.(업데이트 대상변수, 업데이트 값)
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(state)) # 0이 출력됨.
#     for _ in range(3):
#         sess.run(update)
#         print(sess.run(state))

