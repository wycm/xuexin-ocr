#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.AL32UTF8'
import random
import tensorflow.contrib.slim as slim
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from tensorflow.python.ops import control_flow_ops
import label_dict
import sys

stdo = sys.stdout
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = stdo

label_dict = label_dict.label_dict

# 输入参数解析
tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

tf.app.flags.DEFINE_integer('charset_size', len(label_dict), "Choose the first `charset_size` characters only.")
tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 16002, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 100, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 500, "the steps to save")

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', './dataset/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', './dataset/test/', 'the test dataset dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('epoch', 1, 'Number of epoches')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Validation batch size')
tf.app.flags.DEFINE_string('mode', 'inference', 'Running mode. One of {"train", "valid", "test"}')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
FLAGS = tf.app.flags.FLAGS


class DataIterator:
    def __init__(self, data_dir):
        # Set FLAGS.charset_size to a small value if available computation power is limited.
        truncate_path = data_dir + ('%05d' % FLAGS.charset_size)
        print(truncate_path)
        # 遍历训练集所有图像的路径，存储在image_names内
        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            if root < truncate_path:
                self.image_names += [os.path.join(root, file_path) for file_path in file_list]
        random.shuffle(self.image_names)  # 打乱
        # 例如image_name为./train/00001/2.png，提取00001就是其label
        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]

    @property
    def size(self):
        return len(self.labels)

    @staticmethod
    def data_augmentation(images):
        # 镜像变换
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        # 图像亮度变化
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        # 对比度变化
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images

    # batch的生成
    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        # numpy array 转 tensor
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        # 将image_list ,label_list做一个slice处理
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)

        labels = input_queue[1]
        images_content = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
        if aug:
            images = self.data_augmentation(images)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)
        # print 'image_batch', image_batch.get_shape()
        return image_batch, label_batch


def build_graph(top_k):
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')  # dropout打开概率
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='train_flag')
    with tf.device('/gpu:0'):
        # network: conv2d->max_pool2d->conv2d->max_pool2d->conv2d->max_pool2d->conv2d->conv2d->
        # max_pool2d->fully_connected->fully_connected
        # 给slim.conv2d和slim.fully_connected准备了默认参数：batch_norm
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training, 'decay': 0.95}):
            conv3_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv3_1')
            max_pool_1 = slim.max_pool2d(conv3_1, [2, 2], [2, 2], padding='SAME', scope='pool1')
            conv3_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv3_2')
            max_pool_2 = slim.max_pool2d(conv3_2, [2, 2], [2, 2], padding='SAME', scope='pool2')
            conv3_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3_3')
            max_pool_3 = slim.max_pool2d(conv3_3, [2, 2], [2, 2], padding='SAME', scope='pool3')
            conv3_4 = slim.conv2d(max_pool_3, 512, [3, 3], padding='SAME', scope='conv3_4')
            conv3_5 = slim.conv2d(conv3_4, 512, [3, 3], padding='SAME', scope='conv3_5')
            max_pool_4 = slim.max_pool2d(conv3_5, [2, 2], [2, 2], padding='SAME', scope='pool4')

            flatten = slim.flatten(max_pool_4)
            fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024,
                                       activation_fn=tf.nn.relu, scope='fc1')
            logits = slim.fully_connected(slim.dropout(fc1, keep_prob), FLAGS.charset_size, activation_fn=None,
                                          scope='fc2')
        # 因为我们没有做热编码，所以使用sparse_softmax_cross_entropy_with_logits
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

        #         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #         if update_ops:
        #             updates = tf.group(*update_ops)
        #             loss = control_flow_ops.with_dependencies([updates], loss)

        #         global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
        #         optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        #         train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)

        global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            train_op = control_flow_ops.with_dependencies([updates], train_op)

        probabilities = tf.nn.softmax(logits)

        # 绘制loss accuracy曲线
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged_summary_op = tf.summary.merge_all()
        # 返回top k 个预测结果及其概率；返回top K accuracy
        predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
        accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {'images': images,
            'labels': labels,
            'keep_prob': keep_prob,
            'top_k': top_k,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'is_training': is_training,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}


# 获待预测图像文件夹内的图像名字
def get_file_list(path):
    list_name = []
    files = os.listdir(path)
    files.sort()
    for file in files:
        file_path = os.path.join(path, file)
        list_name.append(file_path)
    return list_name


is_build = False
graph = None
saver = None
ckpt = None
sess = None


def inference(name_list):
    global is_build, graph, saver, ckpt, sess
    image_set = []
    # 对每张图进行尺寸标准化和归一化
    for image in name_list:
        temp_image = Image.open(image).convert('L')
        temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
        temp_image = np.asarray(temp_image) / 255.0
        temp_image = temp_image.reshape([-1, 64, 64, 1])
        image_set.append(temp_image)

    # allow_soft_placement 如果你指定的设备不存在，允许TF自动分配设备
    # print('========start inference============')
    # images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
    # Pass a shadow label 0. This label will not affect the computation graph.
    if not is_build:
        # tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        graph = build_graph(top_k=1)
        saver = tf.train.Saver()
        # 自动获取最后一次保存的模型
        current_path = os.getcwd()
        if current_path.endswith('web'):
            # 运行在web环境中
            current_path = current_path + '/xuexin/checkpoint'
        else:
            current_path = './checkpoint'
        ckpt = tf.train.latest_checkpoint(current_path)
        if ckpt:
            saver.restore(sess, ckpt)
            # save_pb()
        is_build = True
    val_list = []
    idx_list = []
    # 预测每一张图
    for item in image_set:
        temp_image = item
        # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
        predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                              feed_dict={graph['images']: temp_image,
                                                         graph['keep_prob']: 1.0,
                                                         graph['is_training']: False})
        val_list.append(predict_val)
        idx_list.append(predict_index)
    # return predict_val, predict_index
    return val_list, idx_list


def pred(path):
    name_list = get_file_list(path)
    # binary_pic(name_list)
    # tmp_name_list = get_file_list('../data/tmp')
    # 将待预测的图片名字列表送入predict()进行预测，得到预测的结果及其index
    final_predict_val, final_predict_index = inference(name_list)
    final_reco_text = []  # 存储最后识别出来的文字串
    # 给出top 3预测，candidate1是概率最高的预测
    pred_val_list = []
    for i in range(len(final_predict_val)):
        candidate1 = final_predict_index[i][0][0]
        # candidate2 = final_predict_index[i][0][1]
        # candidate3 = final_predict_index[i][0][2]
        r = label_dict[int(candidate1)].replace('（', '(').replace("）", ")")
        final_reco_text.append(r)
        print('[the result info] image: {0} predict: {1} ; predict index {2} predict_val {3}'.format(
            name_list[i],
            label_dict[int(candidate1)],
            final_predict_index[i], final_predict_val[i]))
        pred_dict = {'accu': final_predict_val[i]
            , 'shape': cv2.imread(name_list[i]).shape, 'result': r}
        pred_val_list.append(pred_dict)
    # print ('=====================OCR RESULT=======================')
    # 打印出所有识别出来的结果（取top 1）
    result = []
    for i in range(len(final_reco_text)):
        result.append(final_reco_text[i]),
    print(''.join(result))
    return ''.join(result), pred_val_list


def pred_pic(path, base_path='/Users/wangyang/PycharmProjects/py3-venv-demo/img/resources/xuexin_samples/'):
    pic1_path = base_path + path

    for sub_path in os.listdir(pic1_path):
        sub_path = pic1_path + sub_path + '/'
        for i in os.listdir(sub_path):
            if os.path.isdir(sub_path + i):
                pred(sub_path + i)


if __name__ == "__main__":
    # pred('/var/www/tmp/d2d66187-1be2-11ea-a6f7-4c32759549cd/0/0')
    pred(
        '/Users/wangyang/PycharmProjects/py3-venv-demo/img/resources/xuexin_samples/7c2c4b40-166f-11ea-b3c0-a6ee154a7ce6/school1/3/1')
