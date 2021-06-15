import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
from rouge import Rouge

def read_file():
    label_root_dir = "./val_label"
    # 读取批量文件后要写入的文件
    label_set = []
    for file in os.listdir(label_root_dir):
        file_name = label_root_dir + "\\" + file
        with open(file_name, "r", encoding='utf-8') as filein:
         # 按行读取每个文件中的内容
            label = []
            for line in filein:
                label.append(' '.join(line.strip().split()))
            label_set.append(' '.join(label))

    pre_root_dir = "./val_decoded"
    # 读取批量文件后要写入的文件
    pre_set = []
    for file in os.listdir(pre_root_dir):
        file_name = pre_root_dir + "\\" + file
        with open(file_name, "r", encoding='utf-8') as filein:
            # 按行读取每个文件中的内容
            decoded = []
            for line in filein:
                decoded.append(' '.join(line.strip().split()))
            pre_set.append(' '.join(decoded))

    """pre_set2 = ['avd', 'adc']
    label_set2 = ['avd', 'adc']"""
    print(len(label_set))
    print(len(pre_set))

    rouge = Rouge()
    scores = rouge.get_scores(pre_set, label_set, avg=True)
    print("result:/n %d", scores)


if __name__ == '__main__':
    read_file()