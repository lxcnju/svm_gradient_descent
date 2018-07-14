#-*- coding:utf-8 -*-
# ʹ���ݶ��½���� Kernel SVM


import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

train_data_fpath = r"data/svmguide1"      # ѵ�������ļ�������������ݣ�3089������
test_data_fpath = r"data/svmguide1.t"     # ���������ļ���4000����������

train_num = 0                             # ѵ�����ݴ�С
features_num = 0                          # ������Ŀ

batch_size = 32                           # ����С
starter_learning_rate = 0.001             # ��ʼ��ѧϰ��
regularizer = 0.001                       # L2����ϵ��

start_index = 0                           # �����ݵĿ�ʼ����
all_steps = 5000                          # �ܵ�ѵ������

# ��ȡ�����ļ�
def get_data(fpath):
    X = []
    Y = []
    with open(fpath, 'r', encoding = 'utf-8') as fr:
        for line in fr:
            parts = line.split()
            if len(parts) <= 0:
                continue
            label = int(parts[0])
            # ���Ϊ0����-1���
            if label == 0:
                label = -1
            features = [float(p.split(":")[1]) for p in parts[1:]]
            X.append(features)
            Y.append([label])
    X = np.array(X)
    Y = np.array(Y)
    # ��׼��
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, Y

# Ȩ��
def get_weight(shape, regularizer = None):
    w = tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
    # ������
    if regularizer:
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

# ƫ��
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

# ȡ���ݵ���һ��batch
def next_batch(train_x, train_y, batch_size):
    global start_index
    end_index = start_index + batch_size
    if end_index <= train_x.shape[0]:
        batch_x = train_x[start_index:end_index, :]
        batch_y = train_y[start_index:end_index]
        start_index = end_index
    else:
        batch_x = train_x[start_index:, :]
        batch_y = train_y[start_index:]
        start_index = 0
    return batch_x, batch_y

# ��ȡ����
train_x, train_y = get_data(train_data_fpath)
test_x, test_y = get_data(test_data_fpath)
train_num = train_x.shape[0]
features_num = train_x.shape[1]
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


##################### ����� #####################
# ��������
tf.reset_default_graph()

# ���ֵ
all_x = tf.placeholder(tf.float32, [train_num, features_num])     # ���е�ѵ������
bs_x = tf.placeholder(tf.float32, [None, features_num])           # batch x
bs_y = tf.placeholder(tf.float32, [None, 1])                      # batch y

# �����Ȩ��
w_out = get_weight(shape = [train_num, 1], regularizer = regularizer)
b_out = get_bias(shape = [1])

# kernel��
kernel_value = tf.tanh(tf.matmul(bs_x, tf.reshape(all_x, [features_num, train_num])))
print(kernel_value)
# Ԥ���ֵ
y_pre = tf.matmul(tf.multiply(bs_y, kernel_value), w_out) + b_out
print(y_pre)
# ��Ԥ���ֵ��Ϊ1��-1
y_pre_label = tf.where(y_pre > 0, tf.ones(tf.shape(bs_y)), tf.zeros(tf.shape(bs_y)) - 1)
print(y_pre_label)
# ׼ȷ��
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pre_label, bs_y), tf.float32))

# SVMʹ�õ�hinge��ʧ
hinge_loss = tf.reduce_sum(tf.maximum(0.0, 1 - tf.multiply(bs_y, y_pre)))
tf.add_to_collection("losses", hinge_loss)

# �ܵ���ʧ����
loss = tf.add_n(tf.get_collection('losses'))

# ѧϰ��ָ��˥��
global_step = tf.Variable(0, trainable = False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.9, staircase=True)

# ѵ��
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

##################### ��ʼѵ������ #####################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())    # ��ʼ��
    # ��¼ѵ��������ѵ�����Ͳ��Լ��ϵ���ʧ��׼ȷ�ʱ仯
    train_losses = []      # ѵ����loss
    test_losses = []       # ���Լ�loss
    train_accus = []       # ѵ����׼ȷ��
    test_accus = []        # ���Լ�׼ȷ��
    steps_list = []
    for i in range(all_steps):
        batch_x, batch_y = next_batch(train_x, train_y, batch_size)
        _, loss_value = sess.run([train_op, loss], feed_dict = {all_x : train_x, bs_x : batch_x, bs_y : batch_y})
        if i % 200 == 0:
            print("Step i = ", i, "Loss value = ", loss_value)
            train_loss, train_accu = sess.run([loss, accuracy], feed_dict = {all_x : train_x, bs_x : train_x, bs_y : train_y})
            test_loss, test_accu = sess.run([loss, accuracy], feed_dict = {all_x : train_x, bs_x : test_x, bs_y : test_y})
            print(train_loss, train_accu, test_loss, test_accu)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accus.append(train_accu)
            test_accus.append(test_accu)
            steps_list.append(i)
    # ����ѵ��������ʧ��׼ȷ�ʱ仯
    plt.figure()
    plt.plot(steps_list, train_losses, steps_list, test_losses)
    plt.legend(["Train Loss", "Test Loss"])
    plt.xlabel("Train Step")
    plt.ylabel("Loss")
    plt.title("loss_bs_{}_lr_{}_re_{}".format(batch_size, starter_learning_rate, regularizer))
    plt.show()
    
    plt.figure()
    plt.plot(steps_list, train_accus, steps_list, test_accus)
    plt.legend(["Train Accuracy", "Test Accuracy"])
    plt.xlabel("Train Step")
    plt.ylabel("Accuracy")
    plt.title("accuracy_bs_{}_lr_{}_re_{}".format(batch_size, starter_learning_rate, regularizer))
    plt.show()