#! /usr/bin/env python
# -*- coding: utf-8 -*-

import label_dict


def resegment(k):
    return k['shape'][1] <= 10 \
           and k['result'] not in label_dict.small_chars
