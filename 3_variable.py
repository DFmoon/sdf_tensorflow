#-*-coding:utf-8-*-
import tensorflow as tf

state=tf.Variable(0,name='counter')     #定义变量，变量名是counter，变量值是0
print(state.name)

one=tf.constant(1)                      #定义常量

new_value=tf.add(state,one)             #变量加常量
update=tf.assign(state,new_value)       #把变量new_value加载到state中

init=tf.global_variables_initializer()  #初始化所有变量！非常重要，有变量就要初始化

with tf.Session() as sess:
    sess.run(init)                      #激活，必须要有
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))