import os
import random
from time import time

import tensorflow as tf


# 定义超参数
Epoch = 200
img_size = 32
img_channel = 3
class_num = 10
lr_base = 1e-4
weight_decay = 1e-4

on_server = False  # 是否运行在服务器端

batch_size = 128 if on_server is False else 256

train_file_dir = './cifar_img/train/'
test_file_dir = './cifar_img/test/'

buffer_size = 20 * batch_size  # 其实这里即使设置为1效果也不会有太大的区别
seed = 1

# 预先计算好的均值和标准差
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]

train_total_step = (50000 // batch_size + 1) * Epoch


# 构建训练数据集
for _ in range(class_num):
    if _ is 0:
        train_image_list = list(
            map(lambda item: train_file_dir + str(_) + '/' + item, os.listdir(train_file_dir + str(_))))

    else:
        train_image_list += list(
            map(lambda item: train_file_dir + str(_) + '/' + item, os.listdir(train_file_dir + str(_))))


train_label_list = [i // 5000 for i in range(50000)]

# 构建测试数据集
for _ in range(class_num):
    if _ is 0:
        test_image_list = list(
            map(lambda item: test_file_dir + str(_) + '/' + item, os.listdir(test_file_dir + str(_))))

    else:
        test_image_list += list(
            map(lambda item: test_file_dir + str(_) + '/' + item, os.listdir(test_file_dir + str(_))))


test_label_list = [i // 1000 for i in range(10000)]

# 将数据打乱
random.seed(seed)
random.shuffle(train_image_list)
random.seed(seed)
random.shuffle(train_label_list)

random.seed(seed)
random.shuffle(test_image_list)
random.seed(seed)
random.shuffle(test_label_list)

# 构建图像数据归一化矩阵
temp1 = tf.ones((img_size, img_size)) * mean[0]
temp2 = tf.ones((img_size, img_size)) * mean[1]
temp3 = tf.ones((img_size, img_size)) * mean[2]

mean = tf.stack([temp1, temp2, temp3], axis=2)

temp1 = tf.ones((img_size, img_size)) * std[0]
temp2 = tf.ones((img_size, img_size)) * std[1]
temp3 = tf.ones((img_size, img_size)) * std[2]

std = tf.stack([temp1, temp2, temp3], axis=2)


def parse(training):  # 数据解析函数

    def train_parse(image, label):  # 训练数据解析

        # 对图像数据进行各种数据增强
        image = tf.image.decode_png(tf.read_file(image), channels=img_channel)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.05)
        image = tf.image.random_contrast(image, lower=0.83, upper=1.2)

        # 做完数据增强之后不要忘记进行裁剪
        image = tf.clip_by_value(image, 0.0, 1.0)

        # 对图像数据进行归一化
        image = (image - mean) / std

        # 将标签处理成独热码
        label = tf.one_hot(label, depth=class_num)

        return image, label

    def test_parse(image, label):  # 测试数据解析

        image = tf.image.decode_png(tf.read_file(image), channels=img_channel)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # 对图像数据进行归一化
        image = (image - mean) / std

        # 将标签处理成独热码
        label = tf.one_hot(label, depth=class_num)

        return image, label

    if training is True:
        return train_parse
    else:
        return test_parse


# 训练集数据提取
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
train_dataset = train_dataset.map(parse(training=True))
train_dataset = train_dataset.shuffle(buffer_size=buffer_size)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.repeat(Epoch)

train_iterator = train_dataset.make_one_shot_iterator()
train_batch = train_iterator.get_next()

# 测试集数据提取
test_dataset = tf.data.Dataset.from_tensor_slices((test_image_list, test_label_list))
test_dataset = test_dataset.map(parse(training=False))
test_dataset = test_dataset.batch(batch_size)

test_iterator = test_dataset.make_initializable_iterator()
test_batch = test_iterator.get_next()


# 训练集在测试网络时的数据提取
train_eval_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
train_eval_dataset = train_eval_dataset.map(parse(training=False))
train_eval_dataset = train_eval_dataset.batch(batch_size)

train_eval_iterator = train_eval_dataset.make_initializable_iterator()
train_eval_batch = train_eval_iterator.get_next()


class VGGNet():  # VGG13:13 layers

    def __init__(self, sess):
        self.sess = sess

        self.global_step = tf.Variable(0, False)

        self.lr = tf.train.exponential_decay(lr_base, self.global_step,
                                             (50000 // batch_size + 1) * 50, 0.1, True)  # 调整学习率
        self.reg = tf.contrib.layers.l2_regularizer(weight_decay)  # 权重衰减

        self.build_net()

    def build_net(self):  # 构建计算图

        with tf.variable_scope('input'):
            self.img = tf.placeholder(tf.float32, shape=[None, img_size, img_size, img_channel], name='image')
            self.label = tf.placeholder(tf.float32, shape=[None, class_num], name='GT_label')

            self.whether_train = tf.placeholder(tf.bool, name='whether_train')  # BN层和dropout层需要提供的参数

        with tf.variable_scope('Stage_1'):
            with tf.variable_scope('conv_1'):
                conv = tf.layers.conv2d(self.img, 64, 3, padding='same', kernel_regularizer=self.reg,
                                        kernel_initializer=tf.initializers.variance_scaling(2), use_bias=False,
                                        name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, momentum=0.9, name='bn')
                relu = tf.nn.relu(bn, name='relu')

            with tf.variable_scope('conv_2'):
                conv = tf.layers.conv2d(relu, 64, 3, padding='same', kernel_regularizer=self.reg,
                                        kernel_initializer=tf.initializers.variance_scaling(2), use_bias=False,
                                        name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, momentum=0.9, name='bn')
                relu = tf.nn.relu(bn, name='relu')

                pool = tf.layers.max_pooling2d(relu, 3, 2, padding='same', name='pool')

                tf.summary.histogram('after_pooling', pool)

        with tf.variable_scope('Stage_2'):
            with tf.variable_scope('conv_1'):
                conv = tf.layers.conv2d(pool, 128, 3, padding='same', kernel_regularizer=self.reg,
                                        kernel_initializer=tf.initializers.variance_scaling(2), use_bias=False,
                                        name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train,  momentum=0.9, name='bn')
                relu = tf.nn.relu(bn, name='relu')

            with tf.variable_scope('conv_2'):
                conv = tf.layers.conv2d(relu, 128, 3, padding='same', kernel_regularizer=self.reg,
                                        kernel_initializer=tf.initializers.variance_scaling(2), use_bias=False,
                                        name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, momentum=0.9, name='bn')
                relu = tf.nn.relu(bn, name='relu')

                pool = tf.layers.max_pooling2d(relu, 3, 2, padding='same', name='pool')

                tf.summary.histogram('after_pooling', pool)

        with tf.variable_scope('Stage_3'):
            with tf.variable_scope('conv_1'):
                conv = tf.layers.conv2d(pool, 256, 3, padding='same', kernel_regularizer=self.reg,
                                        kernel_initializer=tf.initializers.variance_scaling(2), use_bias=False,
                                        name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, momentum=0.9, name='bn')
                relu = tf.nn.relu(bn, name='relu')

            with tf.variable_scope('conv_2'):
                conv = tf.layers.conv2d(relu, 256, 3, padding='same', kernel_regularizer=self.reg,
                                        kernel_initializer=tf.initializers.variance_scaling(2), use_bias=False,
                                        name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, momentum=0.9, name='bn')
                relu = tf.nn.relu(bn, name='relu')

                pool = tf.layers.max_pooling2d(relu, 3, 2, padding='same', name='pool')

                tf.summary.histogram('after_pooling', pool)

        with tf.variable_scope('Stage_4'):
            with tf.variable_scope('conv_1'):
                conv = tf.layers.conv2d(pool, 512, 3, padding='same', kernel_regularizer=self.reg,
                                        kernel_initializer=tf.initializers.variance_scaling(2), use_bias=False,
                                        name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, momentum=0.9, name='bn')
                relu = tf.nn.relu(bn, name='relu')

            with tf.variable_scope('conv_2'):
                conv = tf.layers.conv2d(relu, 512, 3, padding='same', kernel_regularizer=self.reg,
                                        kernel_initializer=tf.initializers.variance_scaling(2), use_bias=False,
                                        name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, momentum=0.9, name='bn')
                relu = tf.nn.relu(bn, name='relu')

                tf.summary.histogram('after_relu', relu)

        with tf.variable_scope('Stage_5'):
            with tf.variable_scope('conv_1'):
                conv = tf.layers.conv2d(relu, 512, 3, padding='same', kernel_regularizer=self.reg,
                                        kernel_initializer=tf.initializers.variance_scaling(2), use_bias=False,
                                        name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, momentum=0.9, name='bn')
                relu = tf.nn.relu(bn, name='relu')

            with tf.variable_scope('conv_2'):
                conv = tf.layers.conv2d(relu, 512, 3, padding='same', kernel_regularizer=self.reg,
                                        kernel_initializer=tf.initializers.variance_scaling(2), use_bias=False,
                                        name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, momentum=0.9, name='bn')
                relu = tf.nn.relu(bn, name='relu')

                tf.summary.histogram('after_relu', relu)

        with tf.variable_scope('Clf'):
            pool = tf.layers.average_pooling2d(relu, 4, 1, name='pool')
            flatten = tf.layers.flatten(pool, name='flatten')

            self.fc = tf.layers.dense(flatten, class_num,
                                      kernel_initializer=tf.initializers.truncated_normal(stddev=0.1), name='fc')

            tf.summary.histogram('after_fc', self.fc)

        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.fc, name='cross_entropy'))
            self.loss = tf.add(loss, tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

            tf.summary.scalar('loss', self.loss)

        self.train_merge = tf.summary.merge_all()

        # 使用BN的话一定要加上这一步!!!!!!!!!!!!!!!!!!!!
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope('Test_step'):
            self.acc_op = tf.metrics.accuracy(tf.argmax(self.label, axis=1), tf.argmax(self.fc, axis=1))[1]
            self.train_acc_summary = tf.summary.scalar('train_acc', self.acc_op)
            self.test_acc_summary = tf.summary.scalar('test_acc', self.acc_op)


# 开始训练网络
start = time()
os.environ['CUDA_VISIBLE_DEVICES'] = '0' if on_server is False else '1'  # 设置在服务器端运行网络的GPU
writer = tf.summary.FileWriter('./test/')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    # 实例化一个VGG网络, 构建计算图
    net = VGGNet(sess)
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)

    try:
        while True:
            data, label = sess.run(train_batch)
            summary = sess.run(
                                [net.train_merge, net.train_step],
                                feed_dict={net.img: data, net.label: label, net.whether_train: True}
                            )[0]

            if sess.run(net.global_step) % 20 == 0:
                writer.add_summary(summary, sess.run(net.global_step))

                # 测试一下网络在训练集上的精度
                sess.run(tf.local_variables_initializer())
                sess.run(train_eval_iterator.initializer)
                try:
                    while True:
                        data, label = sess.run(train_eval_batch)
                        train_res, summary = net.sess.run(
                            [net.acc_op, net.train_acc_summary],
                            feed_dict={net.img: data, net.label: label, net.whether_train: False}
                        )
                except tf.errors.OutOfRangeError:
                    writer.add_summary(summary, sess.run(net.global_step))

                # 测试一下网络在测试集上的精度
                sess.run(tf.local_variables_initializer())
                sess.run(test_iterator.initializer)
                try:
                    while True:
                        data, label = sess.run(test_batch)
                        test_res, summary = net.sess.run(
                            [net.acc_op, net.test_acc_summary],
                            feed_dict={net.img: data, net.label: label, net.whether_train: False}
                        )
                except tf.errors.OutOfRangeError:
                    writer.add_summary(summary, sess.run(net.global_step))

                print('state:[{:.3f}]%, train_acc:{:.3f}%, test_acc:{:.3f}%, time{:.3f}min'
                      .format(sess.run(net.global_step) / train_total_step, train_res, test_res, (time() - start) / 60))

    except tf.errors.OutOfRangeError:
        print('end')

    writer.close()

# 另一种快速构建网络的方式，代码中构建的是VGG19，相对比上边的程序可以大量减少代码量
# import os
# import random
# from time import time
#
# import tensorflow as tf
#
#
# # 定义超参数
# Epoch = 200
# img_size = 32
# img_channel = 3
# class_num = 10
# lr_base = 1e-4
# weight_decay = 1e-4
#
# on_server = True  # 是否运行在服务器端
#
# batch_size = 128 if on_server is False else 256
#
# train_file_dir = './cifar_img/train/'
# test_file_dir = './cifar_img/test/'
#
# buffer_size = 20 * batch_size  # 其实这里即使设置为1效果也不会有太大的区别
# seed = 1
#
# # 预先计算好的均值和标准差
# mean = [0.4914, 0.4822, 0.4465]
# std = [0.2470, 0.2435, 0.2616]
#
# train_total_step = (50000 // batch_size + 1) * Epoch
#
#
# # 构建训练数据集
# for _ in range(class_num):
#     if _ is 0:
#         train_image_list = list(
#             map(lambda item: train_file_dir + str(_) + '/' + item, os.listdir(train_file_dir + str(_))))
#
#     else:
#         train_image_list += list(
#             map(lambda item: train_file_dir + str(_) + '/' + item, os.listdir(train_file_dir + str(_))))
#
#
# train_label_list = [i // 5000 for i in range(50000)]
#
# # 构建测试数据集
# for _ in range(class_num):
#     if _ is 0:
#         test_image_list = list(
#             map(lambda item: test_file_dir + str(_) + '/' + item, os.listdir(test_file_dir + str(_))))
#
#     else:
#         test_image_list += list(
#             map(lambda item: test_file_dir + str(_) + '/' + item, os.listdir(test_file_dir + str(_))))
#
#
# test_label_list = [i // 1000 for i in range(10000)]
#
# # 将数据打乱
# random.seed(seed)
# random.shuffle(train_image_list)
# random.seed(seed)
# random.shuffle(train_label_list)
#
# random.seed(seed)
# random.shuffle(test_image_list)
# random.seed(seed)
# random.shuffle(test_label_list)
#
# # 构建图像数据归一化矩阵
# temp1 = tf.ones((img_size, img_size)) * mean[0]
# temp2 = tf.ones((img_size, img_size)) * mean[1]
# temp3 = tf.ones((img_size, img_size)) * mean[2]
#
# mean = tf.stack([temp1, temp2, temp3], axis=2)
#
# temp1 = tf.ones((img_size, img_size)) * std[0]
# temp2 = tf.ones((img_size, img_size)) * std[1]
# temp3 = tf.ones((img_size, img_size)) * std[2]
#
# std = tf.stack([temp1, temp2, temp3], axis=2)
#
#
# def parse(training):  # 数据解析函数
#
#     def train_parse(image, label):  # 训练数据解析
#
#         # 对图像数据进行各种数据增强
#         image = tf.image.decode_png(tf.read_file(image), channels=img_channel)
#         image = tf.image.convert_image_dtype(image, tf.float32)
#         image = tf.image.random_flip_left_right(image)
#         image = tf.image.random_brightness(image, 0.05)
#         image = tf.image.random_contrast(image, lower=0.83, upper=1.2)
#
#         # 做完数据增强之后不要忘记进行裁剪
#         image = tf.clip_by_value(image, 0.0, 1.0)
#
#         # 对图像数据进行归一化
#         image = (image - mean) / std
#
#         # 将标签处理成独热码
#         label = tf.one_hot(label, depth=class_num)
#
#         return image, label
#
#     def test_parse(image, label):  # 测试数据解析
#
#         image = tf.image.decode_png(tf.read_file(image), channels=img_channel)
#         image = tf.image.convert_image_dtype(image, tf.float32)
#
#         # 对图像数据进行归一化
#         image = (image - mean) / std
#
#         # 将标签处理成独热码
#         label = tf.one_hot(label, depth=class_num)
#
#         return image, label
#
#     if training is True:
#         return train_parse
#     else:
#         return test_parse
#
#
# # 训练集数据提取
# train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
# train_dataset = train_dataset.map(parse(training=True))
# train_dataset = train_dataset.shuffle(buffer_size=buffer_size)
# train_dataset = train_dataset.batch(batch_size)
# train_dataset = train_dataset.repeat(Epoch)
#
# train_iterator = train_dataset.make_one_shot_iterator()
# train_batch = train_iterator.get_next()
#
# # 测试集数据提取
# test_dataset = tf.data.Dataset.from_tensor_slices((test_image_list, test_label_list))
# test_dataset = test_dataset.map(parse(training=False))
# test_dataset = test_dataset.batch(batch_size)
#
# test_iterator = test_dataset.make_initializable_iterator()
# test_batch = test_iterator.get_next()
#
#
# # 训练集在测试网络时的数据提取
# train_eval_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
# train_eval_dataset = train_eval_dataset.map(parse(training=False))
# train_eval_dataset = train_eval_dataset.batch(batch_size)
#
# train_eval_iterator = train_eval_dataset.make_initializable_iterator()
# train_eval_batch = train_eval_iterator.get_next()
#
#
# class VGGNet():  # VGG19
#
#     def __init__(self, basic_units):
#
#         self.basic_units = basic_units
#
#         self.global_step = tf.Variable(0, False)
#
#         self.lr = tf.train.exponential_decay(lr_base, self.global_step,
#                                              (50000 // batch_size + 1) * 50, 0.1, True)  # 调整学习率
#         self.reg = tf.contrib.layers.l2_regularizer(weight_decay)  # 权重衰减
#
#         self.build_net()
#
#     def make_stage(self, input_tensor, stage, basic_unit, whether_pool=False):
#
#         with tf.variable_scope('Stage_' + str(stage)):
#
#             filter_num = 64 * 2 ** (stage - 1) if stage < 5 else 512
#
#             for index in range(1, basic_unit + 1):
#
#                 with tf.variable_scope('conv_' + str(index)):
#
#                     if index is 1:
#                         input = input_tensor
#                     else:
#                         input = output
#
#                     conv = tf.layers.conv2d(input, filter_num, 3, padding='same', kernel_regularizer=self.reg,
#                                             kernel_initializer=tf.initializers.variance_scaling(2), use_bias=False,
#                                             name='conv')
#                     bn = tf.layers.batch_normalization(conv, training=self.whether_train, momentum=0.9, name='bn')
#                     output = tf.nn.relu(bn, name='relu')
#
#                     if index is basic_unit and whether_pool:
#                         output = tf.layers.max_pooling2d(output, 3, 2, padding='same', name='pool')
#
#             tf.summary.histogram('after_relu' if whether_pool is False else 'after_relu', output)
#         return output
#
#     def build_net(self):  # 构建计算图
#
#         with tf.variable_scope('input'):
#             self.img = tf.placeholder(tf.float32, shape=[None, img_size, img_size, img_channel], name='image')
#             self.label = tf.placeholder(tf.float32, shape=[None, class_num], name='GT_label')
#
#             self.whether_train = tf.placeholder(tf.bool, name='whether_train')  # BN层和dropout层需要提供的参数
#
#         stage = self.make_stage(self.img, 1, self.basic_units[0], True)
#         stage = self.make_stage(stage, 2, self.basic_units[1], True)
#         stage = self.make_stage(stage, 3, self.basic_units[2], True)
#         stage = self.make_stage(stage, 4, self.basic_units[3])
#         stage = self.make_stage(stage, 5, self.basic_units[4])
#
#         with tf.variable_scope('Clf'):
#             pool = tf.layers.average_pooling2d(stage, 4, 1, name='pool')
#             flatten = tf.layers.flatten(pool, name='flatten')
#
#             self.fc = tf.layers.dense(flatten, class_num,
#                                       kernel_initializer=tf.initializers.truncated_normal(stddev=0.1), name='fc')
#
#             tf.summary.histogram('after_fc', self.fc)
#
#         with tf.name_scope('Loss'):
#             loss = tf.reduce_mean(
#                 tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.fc, name='cross_entropy'))
#             self.loss = tf.add(loss, tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
#
#             tf.summary.scalar('loss', self.loss)
#
#         self.train_merge = tf.summary.merge_all()
#
#         # 使用BN的话一定要加上这一步!!!!!!!!!!!!!!!!!!!!
#         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#         with tf.control_dependencies(update_ops):
#             self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
#
#         with tf.name_scope('Test_step'):
#             self.acc_op = tf.metrics.accuracy(tf.argmax(self.label, axis=1), tf.argmax(self.fc, axis=1))[1]
#             self.train_acc_summary = tf.summary.scalar('train_acc', self.acc_op)
#             self.test_acc_summary = tf.summary.scalar('test_acc', self.acc_op)
#
#
# # 开始训练网络
# start = time()
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' if on_server is False else '1'  # 设置在服务器端运行网络的GPU
# writer = tf.summary.FileWriter('./log/')
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
#
# with tf.Session(config=config) as sess:
#
#     # 实例化一个VGG网络, 构建计算图
#     net = VGGNet(basic_units=[2, 2, 4, 4, 4])  # VGG19
#     sess.run(tf.global_variables_initializer())
#     writer.add_graph(sess.graph)
#
#     try:
#         while True:
#             data, label = sess.run(train_batch)
#             summary = sess.run(
#                                 [net.train_merge, net.train_step],
#                                 feed_dict={net.img: data, net.label: label, net.whether_train: True}
#                             )[0]
#
#             if sess.run(net.global_step) % 20 == 0:
#                 writer.add_summary(summary, sess.run(net.global_step))
#
#                 # 测试一下网络在训练集上的精度
#                 sess.run(tf.local_variables_initializer())
#                 sess.run(train_eval_iterator.initializer)
#                 try:
#                     while True:
#                         data, label = sess.run(train_eval_batch)
#                         train_res, summary = sess.run(
#                             [net.acc_op, net.train_acc_summary],
#                             feed_dict={net.img: data, net.label: label, net.whether_train: False}
#                         )
#                 except tf.errors.OutOfRangeError:
#                     writer.add_summary(summary, sess.run(net.global_step))
#
#                 # 测试一下网络在测试集上的精度
#                 sess.run(tf.local_variables_initializer())
#                 sess.run(test_iterator.initializer)
#                 try:
#                     while True:
#                         data, label = sess.run(test_batch)
#                         test_res, summary = sess.run(
#                             [net.acc_op, net.test_acc_summary],
#                             feed_dict={net.img: data, net.label: label, net.whether_train: False}
#                         )
#                 except tf.errors.OutOfRangeError:
#                     writer.add_summary(summary, sess.run(net.global_step))
#
#                 print('state:[{:.3f}]%, train_acc:{:.3f}%, test_acc:{:.3f}%, time{:.3f}min'
#                       .format(sess.run(net.global_step) / train_total_step, train_res, test_res,
#                       (time() - start) / 60))
#
#     except tf.errors.OutOfRangeError:
#         print('end')
#
#     writer.close()

