#-*-coding:utf-8-*-
import tensorflow as tf

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

output=tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[3],input2:[4]}))
    
#通过placeholder可以在sess.run结果时再利用feed_dict字典形式传入参数
#placeholder和feed_dict是绑定的