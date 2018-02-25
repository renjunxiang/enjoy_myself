from pandas import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

sess = tf.InteractiveSession()


def RGB(input_path='D:/tensorflow/picture/5.jpg',
        save_path=None):
    image_raw_data = tf.gfile.FastGFile(input_path, 'rb').read()  # windows下使用'r'会出错无法解码，只能以2进制形式rb读取
    img_data = tf.image.decode_jpeg(image_raw_data)
    picture = 256 - sess.run(img_data) #imshow是0白256黑，要转换

    picture_data = tf.placeholder(dtype=tf.float32)

    kernel_red = tf.constant([
        [[[1.0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    ])
    kernel_green = tf.constant([
        [[[0.0, 0, 0], [0, 1, 0], [0, 0, 0]]]
    ])
    kernel_blue = tf.constant([
        [[[0.0, 0, 0], [0, 0, 0], [0, 0, 1]]]
    ])

    # 中间-四周 凸显边缘
    kernel_edge = tf.constant([
        [[[-1.0, 0, 0], [0, -1.0, 0], [0, .0, -1.0]],
         [[-1.0, 0, 0], [0, -1.0, 0], [0, .0, -1.0]],
         [[-1.0, 0, 0], [0, -1.0, 0], [0, .0, -1.0]]],
        [[[-1.0, 0, 0], [0, -1.0, 0], [0, .0, -1.0]],
         [[8.0, 0, 0], [0, 8, 0], [0, 0, 8]],
         [[-1.0, 0, 0], [0, -1.0, 0], [0, .0, -1.0]]],
        [[[-1.0, 0, 0], [0, -1.0, 0], [0, .0, -1.0]],
         [[-1.0, 0, 0], [0, -1.0, 0], [0, .0, -1.0]],
         [[-1.0, 0, 0], [0, -1.0, 0], [0, .0, -1.0]]]
    ])
    conv2d_red = tf.nn.conv2d([picture_data], kernel_red, strides=[1, 1, 1, 1], padding='SAME')
    conv2d_green = tf.nn.conv2d([picture_data], kernel_green, strides=[1, 1, 1, 1], padding='SAME')
    conv2d_blue = tf.nn.conv2d([picture_data], kernel_blue, strides=[1, 1, 1, 1], padding='SAME')
    conv2d_edge = tf.nn.relu(tf.nn.conv2d([picture_data], kernel_edge, strides=[1, 2, 2, 1], padding='VALID'))

    pictures_red = sess.run(conv2d_red, feed_dict={picture_data: picture})
    pictures_green = sess.run(conv2d_green, feed_dict={picture_data: picture})
    pictures_blue = sess.run(conv2d_blue, feed_dict={picture_data: picture})
    pictures_small = sess.run(conv2d_edge, feed_dict={picture_data: picture})

    fig = plt.figure()
    plt.suptitle('RGB change')

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(img_data.eval())
    plt.title('raw')

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(pictures_red[0])
    plt.title('red')

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(pictures_green[0])
    plt.title('green')

    print(pictures_blue)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(pictures_blue[0])
    plt.title('blue')

    print(pictures_small)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(pictures_small[0])
    plt.title('edge')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    RGB(input_path=os.path.dirname(__file__) + '/picture/6.jpg',
        save_path=os.path.dirname(__file__) + '/picture_transform/6.png')
