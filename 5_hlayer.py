#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))       #赋予权重随机变量矩阵，行列为in_size,out_size
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)                  #定义初始值为0.1的偏置，一行，out_size列的列表
    Wx_plus_b=tf.matmul(inputs,Weights)+biases                      #计算预测值
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

x_data=np.linspace(-1,1,300)[:,np.newaxis]      #添加维度，也就是x有300行
noise=np.random.normal(0,0.05,x_data.shape)     #加入噪点，均值为0，方差为0.05，格式和x_data相同
y_data=np.square(x_data)-0.5+noise

xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

l1=add_layer(xs,1,10,activation_function=tf.nn.relu)                #一个输入，十个输出，即隐藏层设置十个神经元 
prediction=add_layer(l1,10,1,activation_function=None)              #输出层，十个输入，一个输出

loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))         #平均误差

train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)    #设置优化器

init=tf.global_variables_initializer()          #初始化神经网络的所有变量

with tf.Session() as sess:
    sess.run(init)
    plt.ion()           #设置实时打印
    plt.show()
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i%50==0:
            print(i,sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
            prediction_value=sess.run(prediction,feed_dict={xs:x_data,ys:y_data})
            plt.cla()
            plt.scatter(x_data,y_data)
            plt.plot(x_data,prediction_value,'r-',lw=5)
            plt.pause(0.5)
    plt.ioff()          #关闭实时打印
    plt.show()    