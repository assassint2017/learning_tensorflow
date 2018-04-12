import os

import torchvision
import matplotlib.pyplot as plt

num_class = 10

for i in range(num_class):
    os.makedirs('./cifar_img/train/' + str(i))

for i in range(num_class):
    os.makedirs('./cifar_img/test/' + str(i))


# 获取测试集的图像数据
dataset = torchvision.datasets.CIFAR10('./CIFAR10/', False)

index = 0

for data, label in zip(dataset.test_data, dataset.test_labels):

    print(index)

    plt.imsave('./cifar_img/test/' + str(label) + '/' + str(index) + '.png', data)
    index += 1


# 获取训练集的图像数据
dataset = torchvision.datasets.CIFAR10('./CIFAR10/', True)

index = 0

for data, label in zip(dataset.train_data, dataset.train_labels):

    print(index)

    plt.imsave('./cifar_img/train/' + str(label) + '/' + str(index) + '.png', data)
    index += 1
