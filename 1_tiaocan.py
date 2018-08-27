#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#创建数据
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

#创建结构并初始化
Weigths=tf.Variable(tf.random_uniform([1],-1.0,1.0))    #权重初始值，一维，范围-1.0-1.0
biases=tf.Variable(tf.zeros([1]))                       #偏置初始值，一维，初始为0
y=Weigths*x_data+biases                                 #y的预测值
loss=tf.reduce_mean(tf.square(y-y_data))                #损失函数
optimizer=tf.train.GradientDescentOptimizer(0.5)        #优化器，学习效率为0.5
train=optimizer.minimize(loss)
init=tf.global_variables_initializer()                  #初始化神经网络的所有变量

#定义一个指向神经网络的指针
sess=tf.Session()           
sess.run(init)              #激活神经网络

for step in range(201):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(Weigths),sess.run(biases))
     