#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 学信图片切割

from splitter import Splitter
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import chinese_inference
import json
import uuid
import base64
import shutil
import label_dict
import utils

# 0表示带有数字或"（"、"）"等特殊字符
degree_flag = [
    [1, 1],
    [0, 0],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [1, 0]
]
degree_attr = [
    ['truename', 'sex'],
    ['birthdate', 'admissionDate'],
    ['graduationDate', 'schoolName'],
    ['major', 'degreeType'],
    ['', 'learnForm'],
    ['level', 'graduationType'],
    ['', 'certificateNum']
]
school_flag = [
    [1, 1],
    [0, 1],
    [0, 1],
    [1, 1],
    [0, 1],
    [1, 0],
    [0, 0],
    [0, 0],
    [0, 0]
]
school_attr = [
    ['truename', 'sex'],
    ['birthdate', 'nation'],
    ['idcard', 'schoolName'],
    ['level', 'major'],
    ['educationalSystem', 'degreeType'],
    ['learnForm', 'branchName'],
    ['department', 'grade'],
    ['studentId', 'admissionDate'],
    ['departureDate', 'schoolRollStatu']
]

output_path = '/var/www/tmp/'


def segment_and_pred(source_path, print_path, img_type):
    print('Start process pic:' + source_path)
    if 'school' in img_type:
        segment_flag = school_flag
        attr = school_attr
    else:
        segment_flag = degree_flag
        attr = degree_attr
    result = {}
    image_color = cv2.imread(source_path)
    image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    ret, adaptive_threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    ret, at = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    first_column_img = adaptive_threshold[0:image_color.shape[1], 120:350]

    second_column_img = adaptive_threshold[0:image_color.shape[1], 400:659]

    first_column_img_w = first_column_img.shape[0]
    second_column_img_w = second_column_img.shape[0]
    splitter = Splitter()
    # 计算换行内容索引
    horizontal_sum = np.sum(at, axis=1)
    peek_ranges = splitter.extract_peek_ranges_from_array(horizontal_sum)
    last_pr = None
    # 内容换行
    line_feed_index_dict = {}
    # 内容为空
    line_empty_index_dict = {}
    j = 0
    k = 0
    for i, pr in enumerate(peek_ranges):
        if last_pr is not None and pr[0] - last_pr[1] < 15:
            line_feed_index_dict[i] = j
            j += 1
        elif last_pr is not None and pr[0] - last_pr[1] >= 42:
            line_empty_index_dict[i] = k
            k += 1
        last_pr = pr

    # 内容包括换行的行数
    line_feed_count = 0

    # 内容为空的行数
    line_empty_count = 0
    i = 0
    while i < (len(segment_flag) + len(line_feed_index_dict)):
        if i in line_feed_index_dict:
            line_feed_count += 1
        elif i in line_empty_index_dict:
            line_empty_count += 1
            result[attr[i - line_feed_count][0]] = ''
            result[attr[i - line_feed_count][1]] = ''
            i += 1
            continue
        tmp1 = first_column_img[peek_ranges[i - line_empty_count][0]: peek_ranges[i - line_empty_count][1], 0: first_column_img_w - 1]
        splitter.show_img('first image', tmp1)
        kv0_path = print_path + str(i) + '/'
        if not os.path.exists(kv0_path):
            os.makedirs(kv0_path)
        cv2.imwrite(kv0_path + 'kv0.png', tmp1)

        # cv2.waitKey(0)
        if segment_flag[i - line_feed_count][0] == 1:
            min_width = 12
        else:
            min_width = 3
        kv0_print_path = print_path + str(i) + '/0/'
        if os.path.exists(kv0_print_path):
            shutil.rmtree(kv0_print_path)
        if not os.path.exists(kv0_print_path):
            os.makedirs(kv0_print_path)
        splitter.process_by_path(kv0_path + 'kv0.png', kv0_print_path, minimun_range=min_width)
        attr_name = attr[i - line_feed_count][0]
        if attr_name != '':
            pred_result, pred_val_list = chinese_inference.pred(kv0_print_path)
            if resegment(pred_val_list):
                print('re segment:' + kv0_print_path)
                shutil.rmtree(kv0_print_path)
                os.makedirs(kv0_print_path)
                splitter.process_by_path(kv0_path + 'kv0.png', kv0_print_path, minimun_range=min_width,
                                         pred_val_list=pred_val_list)
                pred_result, pred_val_list = chinese_inference.pred(kv0_print_path)
            if attr_name in result:
                # 内容换行
                result[attr_name] = result[attr_name] + pred_result
            else:
                result[attr_name] = pred_result
        tmp2 = second_column_img[peek_ranges[i - line_empty_count][0]: peek_ranges[i - line_empty_count][1], 0: second_column_img_w - 1]
        splitter.show_img('second image', tmp2)
        kv1_path = print_path + str(i) + '/'
        cv2.imwrite(kv1_path + 'kv1.png', tmp2)
        if segment_flag[i - line_feed_count][1] == 1:
            min_width = 12
        else:
            min_width = 3
        kv1_print_path = print_path + str(i) + '/1/'
        if os.path.exists(kv1_print_path):
            shutil.rmtree(kv1_print_path)
        if not os.path.exists(kv1_print_path):
            os.makedirs(kv1_print_path)
        splitter.process_by_path(kv1_path + 'kv1.png', kv1_print_path, minimun_range=min_width)
        attr_name = attr[i - line_feed_count][1]
        if attr_name != '':
            pred_result, pred_val_list = chinese_inference.pred(kv1_print_path)
            if resegment(pred_val_list):
                print('re segment:' + kv1_print_path)
                shutil.rmtree(kv1_print_path)
                os.makedirs(kv1_print_path)
                splitter.process_by_path(kv1_path + 'kv1.png', kv1_print_path, minimun_range=min_width,
                                         pred_val_list=pred_val_list)
                pred_result, pred_val_list = chinese_inference.pred(kv1_print_path)
            if attr_name in result:
                # 内容换行
                result[attr_name] = result[attr_name] + pred_result
            else:
                result[attr_name] = pred_result
        i += 1
    return result


# 是否重新切割
def resegment(pred_result_list):
    for i, k in enumerate(pred_result_list):
        if utils.resegment(k):
            return True
    return False


def compare(attr_name, expect_result, pred_result):
    if '*' in expect_result:
        equals_flag = True
        if len(pred_result) == len(expect_result):
            for i, c in enumerate(expect_result):
                if c != '*' and c != pred_result[i: i + 1]:
                    equals_flag = False
                    break
        else:
            equals_flag = False

        if not equals_flag:
            print('expect ' + attr_name + ':' + expect_result + ', pred ' + attr_name + ':' + pred_result)
    elif pred_result != expect_result and '*' not in expect_result:
        print('expect ' + attr_name + ':' + expect_result + ', pred ' + attr_name + ':' + pred_result)


# todo 处理内容存在换行的情况
def process(base64_str, img_type):
    path = str(uuid.uuid1())

    imgdata = base64.b64decode(base64_str)
    if not os.path.exists(output_path + path):
        os.makedirs(output_path + path)
    print('img path:' + output_path + path)
    img_file_name = output_path + path + '/1.png'
    f = open(img_file_name, 'wb')
    f.write(imgdata)
    f.close()
    r = check_img(img_file_name, img_type)
    if r['code'] != 0:
        return r;
    result = segment_and_pred(img_file_name, output_path + path + '/', img_type)
    r['data'] = result
    return r


# 校验图片size
def check_img(path, img_type):
    i = cv2.imread(path)
    if 'school' in img_type:
        if i.shape[0] >= 378 and (i.shape[0] - 378) % 20 == 0 and i.shape[1] == 660:
            result = {'code': 0, 'desc': ''}
        else:
            result = {'code': -1, 'desc': '图片size非法'}
            print("图片size非法")
    else:
        if i.shape[0] >= 294 and (i.shape[0] - 294) % 20 == 0 and i.shape[1] == 660:
            result = {'code': 0, 'desc': ''}
        else:
            result = {'code': -1, 'desc': '图片size非法'}
            print("图片size非法")
    return result


def main():
    base_path = '/Users/wangyang/PycharmProjects/py3-venv-demo/img/resources/xuexin_samples/'
    for t in os.listdir(base_path):
        result_path = base_path + t + '/result.json'
        f = open(result_path, 'r')
        result = f.read()
        f.close()
        result_json = json.loads(result)
        for pic_name in os.listdir(base_path + t):
            pic_base_path = None
            if pic_name.startswith('degree-info-img-page'):
                match_obj = re.search(r'degree-info-img-page(\d+).', pic_name, re.M | re.I)
                index = match_obj.group(1)
                pic_base_path = base_path + t + '/degree' + index + '/'
                result = segment_and_pred(base_path + t + '/' + pic_name, pic_base_path, 'degree')
                expect_obj = result_json['data']['degreeInfoEntityList'][int(index)]
                for i in range(len(degree_attr)):
                    for j in range(len(degree_attr[i])):
                        attr_name = degree_attr[i][j]
                        if attr_name == '':
                            continue
                        compare(attr_name, expect_obj[attr_name], result[attr_name])

            elif pic_name.startswith('school-roll-img-page'):
                match_obj = re.search(r'school-roll-img-page(\d+).', pic_name, re.M | re.I)
                index = match_obj.group(1)
                pic_base_path = base_path + t + '/school' + index + '/'
                result = segment_and_pred(base_path + t + '/' + pic_name, pic_base_path, 'school')
                expect_obj = result_json['data']['rollInfoEntityList'][int(index)]
                for i in range(len(school_attr)):
                    for j in range(len(school_attr[i])):
                        attr_name = school_attr[i][j]
                        compare(attr_name, expect_obj[attr_name], result[attr_name])


def test():
    # segment_and_pred(
    #     '/Users/wangyang/PycharmProjects/py3-venv-demo/img/resources/xuexin_samples/5011ad60-175b-11ea-b3c0-a6ee154a7ce6/school-roll-img-page0.png'
    #     ,
    #     '/Users/wangyang/PycharmProjects/py3-venv-demo/img/resources/xuexin_samples/5011ad60-175b-11ea-b3c0-a6ee154a7ce6/school0/'
    #     , 'school')
    segment_and_pred(
        '/Users/wangyang/PycharmProjects/py3-venv-demo/img/resources/xuexin_samples/7c2c4b40-166f-11ea-b3c0-a6ee154a7ce6/school-roll-img-page1.png'
        ,
        '/Users/wangyang/PycharmProjects/py3-venv-demo/img/resources/xuexin_samples/7c2c4b40-166f-11ea-b3c0-a6ee154a7ce6/school1/'
        , 'school')
    # segment_and_pred(
    #     '/Users/wangyang/PycharmProjects/py3-venv-demo/img/resources/mult_line/mult_line.png'
    #     ,
    #     '/Users/wangyang/PycharmProjects/py3-venv-demo/img/resources/mult_line/degree0/'
    #     , 'degree')


if __name__ == '__main__':
    # main()
    test()
