import math
from time import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 定义超参数
EPOCH = 100
batch_size = 128
img_size = 28
img_channel = 1
num_class = 10
lr_base = 1e-3
lr_decay = 0.96
weight_decay = 1e-4

train_dataset_size = 60000
test_dataset_size = 10000

train_step_num = math.ceil(train_dataset_size / batch_size)
test_step_num = math.ceil(test_dataset_size / batch_size)

# 定义数据集
mnist = input_data.read_data_sets('./MNIST/', one_hot=True, validation_size=0)


# 定义计算图架构
global_step = tf.Variable(0, False)

lr = lr_base * lr_decay ** (global_step / train_step_num)  # 调整学习率

reg = tf.contrib.layers.l2_regularizer(weight_decay)  # 权重衰减

with tf.variable_scope('Input'):  # 输入层

    inputs = tf.placeholder(tf.float32, shape=[None, img_size * img_size * img_channel], name='image')
    img = tf.reshape(inputs, [-1, img_size, img_size, img_channel])

    labels = tf.placeholder(tf.float32, shape=[None, num_class], name='GT_label')

    bn_train = tf.placeholder(tf.bool, name='bn_eval')

with tf.variable_scope('Conv_layer1'):
    conv_layer_1 = tf.layers.conv2d(img, 6, 5, padding='same', name='Conv', kernel_regularizer=reg)

    bn_layer_1 = tf.layers.batch_normalization(conv_layer_1, training=bn_train, name='BN')
    act_layer_1 = tf.nn.relu(bn_layer_1, name='Relu')
    pool_layer_1 = tf.layers.max_pooling2d(act_layer_1, 2, 2, name='Pooling')

    tf.summary.histogram('Pool', pool_layer_1)

with tf.variable_scope('Conv_layer_2'):
    conv_layer_2 = tf.layers.conv2d(pool_layer_1, 16, 5, padding='valid', name='Conv', kernel_regularizer=reg)
    bn_layer_2 = tf.layers.batch_normalization(conv_layer_2, training=bn_train, name='BN')
    act_layer_2 = tf.nn.relu(bn_layer_2, name='Relu')
    pool_layer_2 = tf.layers.max_pooling2d(act_layer_2, 6, 2, name='Pooling')

    tf.summary.histogram('Pool', pool_layer_2)

flatten = tf.layers.flatten(pool_layer_2, name='flatten')

with tf.variable_scope('Fc'):
    fc_layer1 = tf.layers.dense(flatten, 128, tf.nn.relu, name='layer1', kernel_regularizer=reg)
    fc_layer2 = tf.layers.dense(fc_layer1, 84, tf.nn.relu, name='layer2', kernel_regularizer=reg)
    fc_layer3 = tf.layers.dense(fc_layer2, num_class, name='layer3', kernel_regularizer=reg)

    tf.summary.histogram('layer3', fc_layer3)

with tf.variable_scope('Loss'):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=fc_layer3, name='cross_entropy')
    loss = tf.add(loss, tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
    tf.summary.scalar('loss', tf.reduce_mean(loss))

train_merge = tf.summary.merge_all()

# 使用BN的话一定要加上这一步!!!!!!!!!!!!!!!!!!!!
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

with tf.variable_scope('Test_step'):

    final_acc = tf.metrics.accuracy(tf.argmax(labels, axis=1), tf.argmax(fc_layer3, axis=1))[1]

    acc_summary = tf.summary.scalar('acc', final_acc)

test_merge = tf.summary.merge_all()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# 开始训练网络
start = time()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    writer = tf.summary.FileWriter('./log/')
    writer.add_graph(tf.get_default_graph())

    sess.run(tf.global_variables_initializer())  # 进行全局初始化

    for epoch in range(EPOCH):

        for step in range(train_step_num):

            data, label = mnist.train.next_batch(batch_size)

            sess.run([train_step], feed_dict={inputs: data, labels: label, bn_train: True})

            if step % 20 is 0:
                sess.run(tf.local_variables_initializer())
                summary = sess.run(train_merge, feed_dict={inputs: data, labels: label, bn_train: True})
                writer.add_summary(summary, epoch * train_step_num + step)

                for _ in range(test_step_num):

                    data, label = mnist.test.next_batch(batch_size)

                    sess.run([final_acc], feed_dict={inputs: data, labels: label, bn_train: False})

                    if _ is test_step_num - 1:
                        res, summary = sess.run([final_acc, acc_summary], feed_dict={inputs: data, labels: label, bn_train: False})
                        writer.add_summary(summary, epoch * train_step_num + step)

                print('epoch:{} step:{}, test_acc:{:.3f}, learning_rata:{:.5f}, time{:.3f} min'
                      .format(epoch, step, res, sess.run(lr), (time() - start) / 60))

    writer.close()

