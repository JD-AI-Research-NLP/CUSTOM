#!/usr/bin/env python
# coding: utf-8

# In[240]:


import re
import os
import sys
import pickle


# In[223]:


# 过滤所有括号
def filter_bracket(data):
    res = list()
    brackets = ['(', ')', '）', '（', ',', '\\', '/']
    for i in data:
        flag = False
        for b in brackets:
            if b in i:
                flag=True
                break
        if not flag:
            res.append(i)
    return res
# 最长公共子串
def find_lcsubstr(s1, s2): 
    m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]
    mmax=0   
    p=0  
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                m[i+1][j+1]=m[i][j]+1
                if m[i+1][j+1]>mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return s1[p-mmax:p],mmax  
# 满足标题文档中的4个规则, TODO
# def filter_title_rule(data):
#     black_words = ['ml', '包邮', '满减', '清仓', '特价']
    


# In[241]:


def get_title_from_daren(path):
    text_daren = list()
    with open(path, 'r') as f:
        for i in f:
            text_daren.append(i.strip().split('\t'))
    all_title = list()
    for item in text_daren:
        if len(item)!=18:
          continue
        all_title.append(item[16])
    all_title = [i.split() for i in all_title]
    title_three = list()
    for t in all_title:
        if len(t) == 3:
            title_three.append(t)
    print('three format ratio: ', len(title_three) / len(all_title))
    brand_name = [i[0] for i in title_three]
    sp = [i[1] for i in title_three]
    category3 = [i[2] for i in title_three]
    return brand_name, sp, category3


# In[247]:


def process_brand(brand_name, path_save):
   # 过滤brand name
    brand_name = list(set(brand_name))
    brand_name = [i for i in brand_name if 1 < len(i) <= 6]
    brand_name = filter_bracket(brand_name) 
    print('brand name len: ', len(brand_name))
    with open(os.path.join(path_save, 'title_first_brand'), 'w') as f:
        for i in brand_name:
            f.write(i + '\n')


# In[248]:


def process_category(category3, path_save):
     # 过滤category3
    zh_pattern = re.compile(r'[\u4e00-\u9fa5]+')
    category3 = list(set(category3))
    category3 = [i for i in category3 if 1 < len(i) <= 6]
    category3_eng = [i for i in category3 if not bool(zh_pattern.search(i))]
    category3 = [i for i in category3 if not i in category3_eng]
    category3 = filter_bracket(category3)
    print('category3 len: ', len(category3))
    with open(os.path.join(path_save, 'title_third_category3'), 'w') as f:
        for i in category3:
            f.write(i + '\n')


# In[254]:


def process_sp(sp, path_save):
    # 过滤标题sp
    zh_pattern = re.compile(r'[\u4e00-\u9fa5]+')
    sp = [i for i in sp if 1 < len(i) <= 6]
    sp_eng = [i for i in sp if not bool(zh_pattern.search(i))]
    sp = [i for i in sp if not i in sp_eng]
    sp = filter_bracket(sp)
    freq_sp = dict()
    for i in sp:
        if i in freq_sp:
            freq_sp[i] += 1
        else:
            freq_sp[i] = 1
    sp = list(set(sp))
    print('sp len: ', len(sp))
    with open(os.path.join(path_save, 'title_second_sp'), 'w') as f:
        for i in sp:
            f.write(i + '\n')
    with open(os.path.join(path_save, 'title_second_sp_dict.pkl'), 'wb') as f:
        pickle.dump(obj=freq_sp, file=f)

    with open(os.path.join(path_save, 'title_second_sp_dict'), 'w') as f:
        for k, v in freq_sp.items():
          f.write(k + '\t' + str(v) + '\n')

# In[245]:


# 去除和category3的overlap
# overlap_category3 = []
# for i in sp:
#     buf = i.strip()
#     for j in category3:
#         cat3 = j.strip()
#         if  buf in cat3 or cat3 in buf:
#             overlap_category3.append(buf)
#             break
# sp = [i for i in sp if not i in overlap_category3]


if __name__ == '__main__':
    root = sys.argv[1]
    domain = sys.argv[2]
    folder = sys.argv[3]
    path_daren = os.path.join(root, 'data', domain+'_'+folder, 'train.writing')
    print(path_daren)
    path_save = os.path.join(root, 'data', domain+'_'+folder)
    # path_daren = './zhongbiao/shoubiao_daren'
    # path_save = './zhongbiao'
    print('read daren...')
    brand, sp, category3 = get_title_from_daren(path_daren)
    print('process brand...')
    process_brand(brand, path_save)
    print('process category3...')
    process_category(category3, path_save)
    print('process sp...')
    process_sp(sp, path_save)
    print('finish!')
    

