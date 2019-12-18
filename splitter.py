#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 文字切割 参考:http://chongdata.com/articles/?p=32
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import label_dict
import utils


class Splitter(object):
    def __init__(self):
        print('Create Splitter')

    # 计算投影分割点
    def extract_peek_ranges_from_array(self, array_vals, minimun_val=10, minimun_range=2):
        start_i = None
        end_i = None
        peek_ranges = []
        for i, val in enumerate(array_vals):
            if val > minimun_val and start_i is None:
                start_i = i
            elif val > minimun_val and start_i is not None:
                pass
            elif val < minimun_val and start_i is not None:
                end_i = i
                if end_i - start_i >= minimun_range:
                    peek_ranges.append((start_i, end_i))
                    start_i = None
                end_i = None
            elif val < minimun_val and start_i is None:
                pass
            else:
                raise ValueError("cannot parse this case...")
        if start_i is not None:
            peek_ranges.append((start_i, i + 1))
        return peek_ranges


    def median_split_ranges(self, peek_ranges):
        new_peek_ranges = []
        widthes = []
        for peek_range in peek_ranges:
            w = peek_range[1] - peek_range[0] + 1
            widthes.append(w)
        widthes = np.asarray(widthes)
        median_w = np.median(widthes)
        for i, peek_range in enumerate(peek_ranges):
            num_char = int(round(widthes[i] / median_w, 0))
            if num_char > 1:
                char_w = float(widthes[i] / num_char)
                for i in range(num_char):
                    start_point = peek_range[0] + int(i * char_w)
                    end_point = peek_range[0] + int((i + 1) * char_w)
                    new_peek_ranges.append((start_point, end_point))
            else:
                new_peek_ranges.append(peek_range)
        return new_peek_ranges

    def fill(self, img, i, result_img_path, sub_segment=False):
        flag = False
        left = 0
        right = 0
        top = 0
        bottom = 0
        expect = 30
        if img.shape[0] < expect and img.shape[0] > 10:
            bottom = int((expect - img.shape[0]) / 2)
            top = expect - bottom - img.shape[0]
            flag = True

        if img.shape[1] < expect and img.shape[1] > 10:
            right = int((expect - img.shape[1]) / 2)
            left = expect - right - img.shape[1]
            flag = True

        # if flag:
        #     img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        if img.shape[1] < 8 and not sub_segment:
            # 数字字符，切割顶部和底部的空隙
            # 水平投影
            horizontal_sum = np.sum(img, axis=1)
            peek_ranges = self.extract_peek_ranges_from_array(horizontal_sum)
            img = img[peek_ranges[0][0]:peek_ranges[len(peek_ranges) - 1][1], 0:img.shape[1]]
            self.show_img('sub img', img)

        # resize resize后效果更好
        # img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(result_img_path + str(i) + '.png', img)

        # cv2.imwrite(result_img_path + str(i) + '.png', img)

    def process_by_img(self, image_color, result_img_path, minimun_range=11, sub_segment=False, pred_val_list=[]):
        # new_shape = (image_color.shape[1] * 2, image_color.shape[0] * 2)
        # image_color = cv2.resize(image_color, new_shape)
        if sub_segment and len(image_color.shape) == 3:
            image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

        if len(image_color.shape) == 2:
            adaptive_threshold = image_color
        else:
            # 黑底白字转换白底黑字
            image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
            adaptive_threshold = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                cv2.THRESH_BINARY_INV, 11, 2)

        self.show_img('binary image', adaptive_threshold)

        # 水平投影
        horizontal_sum = np.sum(adaptive_threshold, axis=1)
        # plt.plot(horizontal_sum, range(horizontal_sum.shape[0]))
        # plt.gca().invert_yaxis()
        # plt.show()

        peek_ranges = self.extract_peek_ranges_from_array(horizontal_sum)
        line_seg_adaptive_threshold = np.copy(adaptive_threshold)
        for i, peek_range in enumerate(peek_ranges):
            x = 0
            y = peek_range[0]
            w = line_seg_adaptive_threshold.shape[1]
            h = peek_range[1] - y
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            cv2.rectangle(line_seg_adaptive_threshold, pt1, pt2, 255)
        self.show_img('line image', line_seg_adaptive_threshold)

        vertical_peek_ranges2d = []
        for peek_range in peek_ranges:
            start_y = peek_range[0]
            end_y = peek_range[1]
            line_img = adaptive_threshold[start_y:end_y, :]
            vertical_sum = np.sum(line_img, axis=0)
            vertical_peek_ranges = self.extract_peek_ranges_from_array(
                vertical_sum,
                minimun_val=30,
                minimun_range=minimun_range)
            vertical_peek_ranges2d.append(vertical_peek_ranges)

        # Draw
        counter = 'a'
        color = (0, 0, 255)
        for i, peek_range in enumerate(peek_ranges):
            merge = False
            j = 0
            c = 0
            while j < len(vertical_peek_ranges2d[i]):
                if len(pred_val_list) != 0 and utils.resegment(pred_val_list[j]) and not merge:
                    print('Found a cutting error picture')
                    merge = True
                    counter = chr(ord(counter) + 1)
                    c += 1
                    j += 1
                    continue

                if merge:
                    # 识别率低，可能是一个汉字被分割了，合并汉字
                    x = vertical_peek_ranges2d[i][j - c][0]
                    if vertical_peek_ranges2d[i][j][1] - x <= 10:
                        # 分为3部分的字体，eg:柳
                        c += 1
                        j += 1
                        continue
                    counter = chr(ord(counter) - 1)
                    merge = False
                    c = 0
                else:
                    x = vertical_peek_ranges2d[i][j][0]
                y = peek_range[0]
                w = vertical_peek_ranges2d[i][j][1] - x
                h = peek_range[1] - y
                pt1 = (x, y)
                pt2 = (x + w, y + h)
                cv2.rectangle(image_color, pt1, pt2, color)
                # 切割图片
                sub_img = adaptive_threshold[pt1[1]:pt2[1], pt1[0]:pt2[0]]

                self.fill(sub_img, counter, result_img_path, sub_segment=sub_segment)
                counter = chr(ord(counter) + 1)
                j += 1
        self.show_img('char image', image_color)

    def process_by_path(self, source_img_path, result_img_path, minimun_range=11, sub_segment=False, pred_val_list=[]):
        image_color = cv2.imread(source_img_path)
        self.process_by_img(image_color, result_img_path, minimun_range, sub_segment=sub_segment,
                            pred_val_list=pred_val_list)

    def show_img(self, img_name, img):
        ''
        # cv2.imshow(img_name, img)
        # cv2.waitKey(0)
