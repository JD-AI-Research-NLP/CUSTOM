# 从达人文本中获取每个词的权重信息，目前方案为tfidf

from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import re
import math
import numpy as np
import sys
import os

def get_valid_kb(all_table, all_text):
  pattern = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5^\-+:.]')
  dict_all_key_value = dict() # 存储每个key下所有value集合并去重
  dict_key_value_intext_number = dict() # 存储每个key下value在达人文本中出现的次数
  print('get key value number...')
  for item in all_table:
    split = item.split('##')[1:]
    for buf in split:
      kv = buf.strip().split('$$')
      if len(kv) != 2:
        continue
      if not kv[0].strip() in dict_all_key_value:
        dict_all_key_value[kv[0].strip()] = list()
      value_split = pattern.split(kv[1].strip())
      for value in value_split:
        if value == '':
          continue
        if not value in dict_all_key_value[kv[0].strip()]:
          dict_all_key_value[kv[0].strip()].append(value)
  dict_all_key_value_number = dict()  # 存储每个key下value的个数
  for k, v in dict_all_key_value.items():
    dict_all_key_value_number[k] = len(v)
  print('get key value in text number...')
  for index, text in enumerate(all_text):
    for k, v in dict_all_key_value.items():
      for value in v:
        if value in text:
          dict_key_value_intext_number[k] = dict_key_value_intext_number.setdefault(k, 0) + 1
    if index % 5000 == 0:
      print(index)
  # 按不同阈值进行过滤
  print('start filtering...')
  total_text_num = len(all_text)
  valid_kb = list()
  for k, v in dict_key_value_intext_number.items():
    value_num = dict_all_key_value_number[k]
    if value_num >= 5 and v > (total_text_num / 20):
      valid_kb.append(k)
  return valid_kb

if __name__ == '__main__':
  root = os.path.join(sys.argv[1], 'data')
  domain = sys.argv[2]
  dataset = sys.argv[3]
  folder = sys.argv[4]
  in_folder = domain + "_" + dataset if dataset != 'train' else domain

  path_text = os.path.join(root, domain, 'train.text')
  path_table = os.path.join(root, domain, 'train.table')
  path_stopwords = os.path.join(root, 'cn_stopwords.txt')
  print('reading text...')
  all_text = list()
  with open(path_text, 'r') as f:
      for i in f:
          all_text.append(i.strip())

  print('reading table...')
  all_table = list()
  with open(path_table, 'r') as f:
      for i in f:
          all_table.append(i.strip())

  print('split text...')
  all_text_split = list()
  for i, j in enumerate(all_text):
      all_text_split.append(jieba.lcut(j))
      if i % 10000 == 0:
          print(i)
  
  stop_words = [i.strip() for i in open(path_stopwords, 'r').readlines()]
  print('filter stop words...')
  all_text_split_filter = list()
  for i in all_text_split:
      buf = list()
      buf = [j for j in i if not j in stop_words]
      all_text_split_filter.append(buf)
  
  all_text_split_filter = [' '.join(i) for i in all_text_split_filter]
  tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit(all_text_split_filter)
  tfidf_model2 = tfidf_model.transform(all_text_split_filter)
  word = tfidf_model.get_feature_names()
  weight = tfidf_model2.toarray()
  
  # 计算每个词tfidf时，对词出现的次数进行norm
  print('calculate tfidf...')
  tf_idf_plus_values = [0.0] * len(word)
  tf_idf_word_count = [0] * len(word)
  for i in range(len(weight)):
      values_plus = np.array(tf_idf_plus_values) + np.array(weight[i])
      word_count =  np.array(tf_idf_word_count) + np.where(np.array(weight[i])>0, 1, 0)
      tf_idf_plus_values = values_plus
      tf_idf_word_count = word_count
      if i % 5000 == 0:
          print(i)
  tf_idf_plus_values_norm = tf_idf_plus_values / tf_idf_word_count
  all_word_tf_idf_plus = dict(zip(word, tf_idf_plus_values_norm))
  all_word_tf_idf_plus_nonorm = dict(zip(word, tf_idf_plus_values))

  print('saving...')
  print(len(all_word_tf_idf_plus))
  with open(os.path.join(root, 'tf_idf_nonorm'), 'w') as f:
    for k, v in all_word_tf_idf_plus_nonorm.items():
      f.write(k + '\t' + '{:.5f}'.format(v) + '\n')    
  
  with open(os.path.join(root, 'tf_idf_norm'), 'w') as f:
    for k, v in all_word_tf_idf_plus.items():
      f.write(k + '\t' + '{:.5f}'.format(v) + '\n')    

  print('processing kb...')
  valid_kb = get_valid_kb(all_table, all_text)
  with open(os.path.join(root, 'valid_kb'), 'w') as f:
    for k in valid_kb:
      f.write(k + '\n')
