#!/usr/bin/env python
# coding: utf-8

# In[31]:


import re
import os
import sys
import math
from random import choice
import pickle
import numpy as np
pattern = r'^[\u4e00-\u9fa5]+$'
zh_pattern = re.compile(r'[\u4e00-\u9fa5]+')
root = './shouji'
back_up_word = ['日式','欧式','家用进口','家用无线','玻璃','男士','女士','陶瓷','女士商务','男士商务','创意', '个性', '304','一体','家用','北欧',                '全自动','成人','64G','64g','128g','128G','金色','红色','黑色','紫色','白色','绿色','原装','金色','银色','联通电信', '蓝色', '卡其色']
phone_category3_erji = ['蓝牙耳机','无线耳机','耳麦','耳机']
phone_category3 = ['拍照手机','游戏手机','老人手机','老人机','智能手机','AI手机','全面屏手机','全网通手机','美颜手机']
other_category = ['手机壳', '移动电源', '创意配件', '充电器', '数据线', '模拟对讲机', '办号卡', '手机贴膜', '手机支架', '苹果周边', '拍照配件']
max_length = 18


# In[32]:


def str_match(title, all_name):
    category3 = []
    for i in all_name:
        if i.strip() in title and title[title.index(i.strip())-3:title.index(i.strip())]!='不支持':
            category3.append(i.strip())
            #category3.sort(key=lambda x: len(x))
    category3.sort(key=lambda x: title.index(x))
    for i in category3:
        if '套装' in i or '套餐' in i:
            return [i]
    return category3
def word_overlap(prodName, cat3):
    cnt = 0
    for ch in prodName:
        if ch in cat3:
            cnt += 1
    return math.log((len(prodName)+1)/(cnt+1))*(cnt/len(cat3))
def product_name_selection(title, all_name, cat3):
    category3 = []
    for i in all_name:
        if i.strip() in title:
            category3.append(i.strip())
            #category3.sort(key=lambda x: len(x))
    category3.sort(key=lambda x: word_overlap(x, cat3), reverse=True)
    return category3
def str_match_return_length_dict(a, b):
    match = {}
    for i in b:
        if i.strip() in a and '时尚' not in i.strip():
            match.setdefault(str(len(i.strip())), []).append(i.strip())
    return match
def all_are_eng(name):
    return True if not zh_pattern.search(name) else False
def find_lcsubstr(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax
def is_chinese(uchar):
    if '\u4e00' <= uchar <= '\u9fff':
        return True
    else:
        return False
def is_number(uchar):
    if '\u0030' <= uchar and uchar <= '\u0039':
        return True
    else:
        return False
def is_english(uchar):
    if ('\u0041' <= uchar <= '\u005a') or ('\u0061' <= uchar <= '\u007a'):
        return True
    else:
        return False
def is_other(uchar):
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False
def contain_number(uchar):
    flag = False
    for i in uchar:
        if is_number(i):
            flag = True
            break
    return flag
def measure_length(a):
    length = 0.0
    for i in a:
        if is_english(i):
            length += 0.5
        elif is_chinese(i):
            length += 1
    return length

def find_phone_brand(brand_name_buf, title, all_brand):
    flag = True
    brand = ''
    for i in all_brand:
        if i.strip() in title:
            brand = i.strip()
            break
    index = title.index(brand)
    brand_length = len(brand)
    if len(brand) > 0 and is_english(brand[-1]):
        return brand_name_buf
    cnt = 0
    for i in range(index + brand_length, len(title)):
        if is_chinese(title[i]):
            break
        elif is_english(title[i]) or is_number(title[i]):
            cnt+=1
            brand += title[i]
        if(cnt > 8):
            return brand_name_buf
    if '苹果' in brand and '苹果' not in brand_name_buf:
        brand = brand_name_back
    return brand

def find_phone_category3(title, category3_kb):
    category3 = '手机'
    if category3_kb in other_category:
        return category3_kb
    if '耳' in category3_kb:
        for j in phone_category3_erji:
            if j in title :# or i in kb_info:
                category3 = j
                break
    else:
         for j in phone_category3:
            if j in title:# or i in kb_info:
                category3 = j
                break
    return category3


# In[33]:


def find_middle(text, hot_set, hot_set_dict, brand_name, category3):
    middle = ''
    length_middle = max_length - measure_length(brand_name + category3)
    if int(length_middle) <= 1:
        return middle
    hot_middle = str_match(text, hot_set)
    hot_middle_copy = hot_middle[:]
    for i in hot_middle:
        if i in category3 or category3 in i or i in brand_name or brand_name in i:
            hot_middle_copy.remove(i)
            continue
        if (find_lcsubstr(i, category3)[1] > 1 and i.replace(find_lcsubstr(i, category3)[0], '')) or i[-1] == category3[
            0]:
            hot_middle_copy.remove(i)
            # hot_middle_copy.append(i.replace(find_lcsubstr(i, category3)[0],''))
    hot_middle_copy = list(set(hot_middle_copy))
    hot_middle_copy = [i for i in hot_middle_copy if len(i) <= length_middle]
    hot_middle_dict = str_match_return_length_dict(text, hot_middle_copy)
    for i in range(int(math.ceil(length_middle))):
        if hot_middle_dict.get(str(int(length_middle - i))):
            middle = hot_middle_dict.get(str(int(length_middle - i)))
            middle.sort(key=lambda x: hot_set_dict[x])
            middle = middle[0]
            break
    if len(middle) <= 1:
        return ''
    if middle[-1] == '电' and middle[-2] != '家' and middle[-2] != '充' and middle[-2] != '断' and middle[-2] != '插' and         middle[-2] != '省' or middle[-1] == '小' or middle[-1] == '大':
        middle = middle.replace(middle[-1], '')
        middle = middle.replace('小', '')
        middle = middle.replace('大', '')
    if middle[-1] == '抽':
        middle = middle.replace(middle[-1], '')
    return middle

def back_up_in(text, back_up_word):
    flag = False
    word = ''
    for i in back_up_word:
        if i in text:
            flag = True
            word = i
            break
    return flag, word


# In[34]:


# 首先获取brand name

def get_brand(title_kb, test_kb, all_brand, sp_rule=None):
    list_brand_name = list()
    for i in range(len(title_kb)):
        title = title_kb[i]
        kvs = test_kb[i].split('##')
        brand_name_buf = kvs[2].split('$$')[1].strip()
        if sp_rule:
            brand_name = sp_rule(brand_name_buf, title, all_brand)
        else:
            brand_name = brand_name_buf
        list_brand_name.append(brand_name)
    return list_brand_name
        


# In[59]:


def get_category3(title_kb, test_kb, all_name, sp_rule=None):
    list_category3 = list()
    for i in range(len(title_kb)):
        title = title_kb[i]
        kvs = test_kb[i].split('##')
        category3_buf = kvs[6].split('$$')[1].strip()
        if('/' in category3_buf):
            category3_buf = category3_buf.split('/')[0]
        category3_buf2 = product_name_selection(title, all_name, category3_buf) 
        category3 = ''
        #if category3_buf2:
        #    category3 = category3_buf2[0]
        #else:
        category3 = category3_buf
        if sp_rule:
            category3 = sp_rule(title, category3_buf)
        list_category3.append(category3)
    return list_category3


# In[60]:

def get_middle(title_kb, test_kb, hotset, hot_set_dict, list_brand_name, list_category3, sp_rule=None):
    list_middle = list()
    cnt_middle_null = 0
    for i in range(len(title_kb)):
        title = title_kb[i]
        kvs = test_kb[i].split('##')
        tmp = [kv.split(' $$ ')[1] for kv in kvs if '特性 $$ ' in kv or '特征 $$ ' in kv or '热点 $$ ' in kv]
        all_info = title + ' ' + ' '.join(tmp)
        kb_info = ''.join(kvs[1:])
        brand_name = list_brand_name[i]
        category3 = list_category3[i]
        middle = 'NULL'
        if find_middle(all_info, hot_set, hot_set_dict, brand_name, category3):
            middle = find_middle(all_info, hot_set, hot_set_dict, brand_name, category3)
        elif find_middle(kb_info, hot_set, hot_set_dict, brand_name, category3):
            middle = find_middle(kb_info, hot_set, hot_set_dict, brand_name, category3)
        elif back_up_in(title + kb_info, back_up_word)[0] and             len(back_up_in(title + kb_info, back_up_word)[1]) <= int(max_length - len(brand_name) - len(category3)):
            middle = back_up_in(title + kb_info, back_up_word)[1]
        else:
            cnt_middle_null += 1
            print(brand_name, ' ', category3)
        list_middle.append(middle)
    print('middle null length: ', cnt_middle_null)
    return list_middle
        


# In[61]:


def read_data(root, path_table, path_dataset):
    with open(os.path.join(path_table, path_dataset+'.table'), 'r', encoding='utf-8') as f:
        test_kb = f.readlines()
    title_kb = [i.split('##')[0].split('$$')[1] for i in test_kb]
    with open(os.path.join(root, 'title_first_brand'),'r', encoding = 'utf-8') as f:
        all_brand = f.readlines()
    with open(os.path.join(root, 'title_second_sp'),'r',encoding = 'utf-8') as f:
        hot_set = f.readlines()
    with open(os.path.join(root, 'title_second_sp_dict.pkl'), 'rb') as f:
        hot_set_dict = pickle.load(f)
    with open(os.path.join(root, 'title_third_category3'), 'r', encoding='utf-8') as f:
        all_name = f.readlines()
    return test_kb, title_kb, all_brand, hot_set, hot_set_dict, all_name


# In[62]:


if __name__ == '__main__':
    data_final = list()
    root = sys.argv[1]
    domain = sys.argv[2]
    folder = sys.argv[3]
    dataset = sys.argv[4]
    path_root = os.path.join(root, 'data', domain+'_'+"rawData")
    path_table = os.path.join(root, 'data', domain+'_'+dataset)
    path_dataset = dataset
    # path_root = '../生产结果/电脑/xuanpin6/tt'
    # path_table = '../生产结果/电脑/xuanpin6/'
    # path_dataset = 'xuanpin6' # 选品名称
    print('read data')
    test_kb, title_kb, all_brand, hot_set, hot_set_dict, all_name = read_data(path_root, path_table, path_dataset) 
    print('get brand')
    list_brand_name = get_brand(title_kb, test_kb, all_brand)
    print('get category3')
    list_category3 = get_category3(title_kb, test_kb, all_name, find_phone_category3)
    print('get middle')
    list_middle = get_middle(title_kb, test_kb, hot_set, hot_set_dict, list_brand_name, list_category3)
    for i in range(len(list_middle)):
        data_final.append(list_brand_name[i] + '\x20' + list_middle[i] + '\x20' + list_category3[i] + '\n')
    with open(os.path.join(path_table, path_dataset+'.title'), 'w', encoding='utf-8') as f:
        for item in data_final:
            f.write(item)
    print('finish')


