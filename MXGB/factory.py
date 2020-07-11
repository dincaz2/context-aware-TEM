__author__ = "xiangwang"

import os
import codecs
import re


def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def uni2str(unicode_str):
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def delMultiChar(inputString, chars):
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString

def merge2List(list1, list2):
    result_list = list1
    for item in list2:
        result_list.append(item)
    return result_list