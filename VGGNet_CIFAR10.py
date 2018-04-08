import os
from time import time
import tensorflow as tf


# 定义超参数
Epoch = 200
batch_size = 64
img_size = 32
img_channel = 3
class_num = 10
lr_base = 1e-4
weight_decay = 1e-5


train_file_dir = './cifar_img/train/'
test_file_dir = './cifar_img/test/'

train_total_step = (50000 // batch_size + 1) * Epoch
test_total_step = 10000 // batch_size + 1


# 构建数据集
for _ in range(class_num):
    if _ is 0:
        train_image_list = list(
            map(lambda item: train_file_dir + str(_) + '/' + item, os.listdir(train_file_dir + str(_))))

    else:
        train_image_list += list(
            map(lambda item: train_file_dir + str(_) + '/' + item, os.listdir(train_file_dir + str(_))))


train_label_list = [i // 5000 for i in range(50000)]

for _ in range(class_num):
    if _ is 0:
        test_image_list = list(
            map(lambda item: test_file_dir + str(_) + '/' + item, os.listdir(test_file_dir + str(_))))

    else:
        test_image_list += list(
            map(lambda item: test_file_dir + str(_) + '/' + item, os.listdir(test_file_dir + str(_))))


test_label_list = [i // 1000 for i in range(10000)]


def parse(image, label):  # 数据解析

    # 对图像数据进行各种数据增强
    image = tf.image.decode_png(tf.read_file(image), channels=img_channel)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, lower=1.0, upper=1.5)

    # 做完数据增强之后不要忘记进行裁剪
    image = tf.clip_by_value(image, 0.0, 1.0)

    # 将标签处理成独热码
    label = tf.one_hot(label, depth=class_num)

    return image, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
train_dataset = train_dataset.map(parse)
train_dataset = train_dataset.shuffle(buffer_size=10000)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.repeat(Epoch)

train_iterator = train_dataset.make_one_shot_iterator()
train_batch = train_iterator.get_next()


test_dataset = tf.data.Dataset.from_tensor_slices((test_image_list, test_label_list))
test_dataset = test_dataset.map(parse)
test_dataset = test_dataset.batch(batch_size)

test_iterator = test_dataset.make_initializable_iterator()
test_batch = test_iterator.get_next()


class VGGNet():  # VGG13:13 layers

    def __init__(self, sess):
        self.sess = sess

        self.global_step = tf.Variable(0, False)

        self.lr = tf.train.exponential_decay(lr_base, self.global_step,
                                             (50000 // batch_size + 1) * 25, 0.1, True)  # 调整学习率
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
                                         kernel_initializer=tf.initializers.variance_scaling(2), name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, name='bn')
                relu = tf.nn.relu(bn, name='relu')

            with tf.variable_scope('conv_2'):
                conv = tf.layers.conv2d(relu, 64, 3, padding='same', kernel_regularizer=self.reg,
                                         kernel_initializer=tf.initializers.variance_scaling(2), name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, name='bn')
                relu = tf.nn.relu(bn, name='relu')

                pool = tf.layers.max_pooling2d(relu, 2, 2, name='pool')

                tf.summary.histogram('after_pooling', pool)

        with tf.variable_scope('Stage_2'):
            with tf.variable_scope('conv_1'):
                conv = tf.layers.conv2d(pool, 128, 3, padding='same', kernel_regularizer=self.reg,
                                         kernel_initializer=tf.initializers.variance_scaling(2), name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, name='bn')
                relu = tf.nn.relu(bn, name='relu')

            with tf.variable_scope('conv_2'):
                conv = tf.layers.conv2d(relu, 128, 3, padding='same', kernel_regularizer=self.reg,
                                         kernel_initializer=tf.initializers.variance_scaling(2), name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, name='bn')
                relu = tf.nn.relu(bn, name='relu')

                pool = tf.layers.max_pooling2d(relu, 2, 2, name='pool')

                tf.summary.histogram('after_pooling', pool)

        with tf.variable_scope('Stage_3'):
            with tf.variable_scope('conv_1'):
                conv = tf.layers.conv2d(pool, 256, 3, padding='same', kernel_regularizer=self.reg,
                                         kernel_initializer=tf.initializers.variance_scaling(2), name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, name='bn')
                relu = tf.nn.relu(bn, name='relu')

            with tf.variable_scope('conv_2'):
                conv = tf.layers.conv2d(relu, 256, 3, padding='same', kernel_regularizer=self.reg,
                                         kernel_initializer=tf.initializers.variance_scaling(2), name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, name='bn')
                relu = tf.nn.relu(bn, name='relu')

                pool = tf.layers.max_pooling2d(relu, 2, 2, name='pool')

                tf.summary.histogram('after_pooling', pool)

        with tf.variable_scope('Stage_4'):
            with tf.variable_scope('conv_1'):
                conv = tf.layers.conv2d(pool, 512, 3, padding='same', kernel_regularizer=self.reg,
                                         kernel_initializer=tf.initializers.variance_scaling(2), name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, name='bn')
                relu = tf.nn.relu(bn, name='relu')

            with tf.variable_scope('conv_2'):
                conv = tf.layers.conv2d(relu, 512, 3, padding='same', kernel_regularizer=self.reg,
                                         kernel_initializer=tf.initializers.variance_scaling(2), name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, name='bn')
                relu = tf.nn.relu(bn, name='relu')

                tf.summary.histogram('after_relu', relu)

        with tf.variable_scope('Stage_5'):
            with tf.variable_scope('conv_1'):
                conv = tf.layers.conv2d(relu, 512, 3, padding='same', kernel_regularizer=self.reg,
                                         kernel_initializer=tf.initializers.variance_scaling(2), name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, name='bn')
                relu = tf.nn.relu(bn, name='relu')

            with tf.variable_scope('conv_2'):
                conv = tf.layers.conv2d(relu, 512, 3, padding='same', kernel_regularizer=self.reg,
                                          kernel_initializer=tf.initializers.variance_scaling(2), name='conv')
                bn = tf.layers.batch_normalization(conv, training=self.whether_train, name='bn')
                relu = tf.nn.relu(bn, name='relu')

                tf.summary.histogram('after_relu', relu)

        with tf.variable_scope('Clf'):
            pool = tf.layers.max_pooling2d(relu, 4, 1, name='pool')
            flatten = tf.layers.flatten(pool, name='flatten')

            self.fc = tf.layers.dense(flatten, class_num, name='fc')

            tf.summary.histogram('after_fc', self.fc)

        with tf.variable_scope('Loss'):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.fc, name='cross_entropy'))
            self.loss = tf.add(loss, tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

            tf.summary.scalar('loss', self.loss)

        self.train_merge = tf.summary.merge_all()

        # 使用BN的话一定要加上这一步!!!!!!!!!!!!!!!!!!!!
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(lr_base).minimize(self.loss, global_step=self.global_step)

        with tf.variable_scope('Test_step'):
            self.acc_op = tf.metrics.accuracy(tf.argmax(self.label, axis=1), tf.argmax(self.fc, axis=1))[1]
            self.acc_summary = tf.summary.scalar('acc', self.acc_op)

    def train(self, images, labels):

        return self.sess.run(
            [self.train_merge, self.train_step],
            feed_dict={self.img: images, self.label: labels, self.whether_train: True}
        )[0]

    def test(self, images, labels):

        return self.sess.run(
            [self.acc_op, self.acc_summary],
            feed_dict={self.img: images, self.label: labels, self.whether_train: False}
        )


# 开始训练网络
start = time()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置在服务器端运行网络的GPU
writer = tf.summary.FileWriter('./log/')
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
            summary = net.train(data, label)

            if int(sess.run(net.global_step)) % 20 is 0:
                writer.add_summary(summary, sess.run(net.global_step))
                sess.run(tf.local_variables_initializer())
                sess.run(test_iterator.initializer)

                for _ in range(test_total_step):
                    data, label = sess.run(test_batch)
                    net.test(data, label)

                    if _ is test_total_step - 1:
                        res, summary = net.test(data, label)
                        writer.add_summary(summary, sess.run(net.global_step))

                print('state:[{:.3f}]%, test_acc:{:.3f}, time{:.3f} min'
                      .format(sess.run(net.global_step) / train_total_step, res, (time() - start) / 60))

    except tf.errors.OutOfRangeError:
        pass

    writer.close()

