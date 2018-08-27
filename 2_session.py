#-*-coding:utf-8-*-
import tensorflow as tf

matrix1=tf.constant([[3,3]])
matrix2=tf.constant([[2],[2]])

product=tf.matmul(matrix1,matrix2)      #矩阵的乘法

##session的两种使用形式
##1
#sess=tf.Session()
#result=sess.run(product)        #每run一次，才会执行一次
#print(result)
#sess.close()

#2，免关闭
with tf.Session() as sess:
    result=sess.run(product)
    print(result)