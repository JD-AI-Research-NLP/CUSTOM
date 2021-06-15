import sys
import re
import jieba
import pdb
import os

def get_source_dict(path, path_daren, path_title_sp, thirdIDs, source_dict):
  print('reading cat3 names')
  cat3_names = {}
  with open(thirdIDs, 'r', encoding='utf-8') as f:
    for catid in f:
      if catid not in cat3_names:
        cat3_names[catid] = 1

  print('reading kb...')
  all_kb = list()
  with open(path, 'r') as f1:
    for data in f1:
      sp = data.strip().split('##')[7:]
      for buf in sp:
        parts = buf.strip().split(' $$ ')
        if len(parts) != 2:
          continue
        sp2 = parts[1].strip()
        if '/' in sp2:
          sp2 = sp2.split('/')[0]
        if '-' in sp2:
          sp2 = sp2.split('-')[1]
        if len(sp2) < 2 or len(sp2) > 8:
          continue
        zh_pattern = re.compile(r'[\u4e00-\u9fa5]+')
        if not bool(zh_pattern.search(sp2)):
          continue
        all_kb.append(sp2.lower())

  all_title_sp = list()
  with open(path_title_sp, 'r') as f:
    for i in f:
      all_title_sp.append(i.strip())

  all_kb = all_kb + all_title_sp
  new_all_kb = list()
  for item in all_kb:
    if item in cat3_names:
      continue
    new_all_kb.append(item)

  all_kb = list(set(new_all_kb))
  all_dic = dict()
  print('find in daren...')
  with open(path_daren, 'r') as f:
    for i, data in enumerate(f):
      if i %  1000 == 0:
        print(i)
      for buf in all_kb:
        if buf in data:
          if buf  in  all_dic:
            all_dic[buf] += 1
          else:
            all_dic[buf] = 1

  #all_dic = list(set(all_dic))
  print('saving...')
  sort = sorted(all_dic.items(), key=lambda x: x[1])
  dic_sort = dict(sort)
  print(len(all_dic))
  with open(source_dict, 'w') as f:
    for k, v in all_dic.items():
      if v >= 100:
        f.write(k+'\n')

        
def get_source_dict_label(train_text, source_dict, source_dict_label):
  def read_source_dict(source_dict_file):
      source = []
      with open(source_dict_file, 'r', encoding='utf-8') as f:
          for line in f:
              line = line.strip()
              source.append(line)
      return source
#             kvs = [kv.split(' $$ ') for kv in line.split(' ## ') if len(kv.split(' $$ ')) == 2]
  def find_value_tokens(text_file, source_dict_file):
      value2subsent = {}
      tok2tf = {}
      source_dict = read_source_dict(source_dict_file)
      cnt = 0
      number = 0.0
      total = 0.0
      with open(text_file, 'r', encoding='utf-8') as f, open(source_dict_label, 'w', encoding='utf-8') as w, open(source_dict_file + '.token', 'w', encoding='utf-8') as w1:
          for line in f:
              line = line.strip()
              subsents = [_.strip() for _ in re.split(r'[。；！？]', line) if _.strip() != '']
  #             pdb.set_trace()
              for subsent in subsents:
                  tokens = [_.strip() for _ in jieba.cut(subsent) if _.strip() != '']
                  flag = False
                  for value in source_dict:
                      if value in subsent:
                          flag = True
                          if value not in value2subsent:
                              value2subsent[value] = []
                          value2subsent[value].append(tokens)
                          for tok in tokens:
                              if tok not in tok2tf:
                                  tok2tf[tok] = 0
                              tok2tf[tok] += 1
                  if flag:
                      number += 1
              total += len(subsents)
              cnt += 1
              if cnt % 10000 == 0:
                  print(cnt)

          sortedValue2Subsent = sorted(value2subsent.items(), key=lambda x: len(x[1]), reverse=True)
          sortedTok2tf = sorted(tok2tf.items(), key=lambda x: x[1], reverse=True)
          toks = {}
          for k, v in sortedValue2Subsent:
              k_toks = set()
              for tokens in v:
                  for tok in tokens:
                      if tok.endswith(k) or k.endswith(tok) or tok.startswith(k) or k.startswith(tok):
                          toks[tok] = 1
                          k_toks.add(tok)
              w.write(str(k) + ' _|||_ ' + ' '.join(k_toks) + '\n')
          for k, v in sortedTok2tf:
              if k not in toks:
                  w1.write('0\t' + str(k) + ' _|||_ ' + str(v) + '\n')
              else:
                  w1.write('1\t' + str(k) + ' _|||_ ' + str(v) + '\n')


          print('Coverage: %f/%f=%f'%(number, total, number/total))
    
  find_value_tokens(train_text, source_dict)

  
def get_source_dict_copy(infile, source_dict, source_dict_copy):
  with open(infile, 'r', encoding='utf-8') as f, open(source_dict, 'w', encoding='utf-8') as w, open(source_dict_copy, 'w', encoding='utf-8') as w1:
      a = set()
      for line in f:
          line = line.strip()
          parts = line.split(' _|||_ ')
          if(len(parts) != 2):
              continue
          w.write(parts[0].strip()+'\n')
          for i in parts[1].split(' '):
              if i.strip() != '':
                  a.add(i)
      for item in a:
          w1.write(item + '\n')
        
if __name__ == '__main__':
  root = sys.argv[1]
  domain = sys.argv[2]
  folder = sys.argv[3]
  
  train_table = os.path.join(root, 'data', domain, 'train.table')
  train_text = os.path.join(root, 'data', domain, 'train.text')
  title_sp = os.path.join(root, 'data', domain + '_' + folder, 'title_second_sp')
  third_category = os.path.join(root, 'data', domain + '_' + folder, 'title_third_category3')
  source_dict = os.path.join(root, 'data', 'trigram', domain, 'source.dict')
  source_dict_label = os.path.join(root, 'data', 'trigram', domain, 'source.dict.copy.tobelabel')
  source_dict_copy = os.path.join(root, 'data', 'trigram', domain, 'source.dict.copy')

  # step1: 获取高亮卖点词表：然后需要人工过滤
#   get_source_dict(train_table, train_text, title_sp, third_category, source_dict)
  
  # step2: 获取高亮词表的分词；然后人工过滤，滤掉比较general的词
#   get_source_dict_label(train_text, source_dict, source_dict_label)
  
  # step3: 根据标注的source.dict.copy.tobelabel生成source.dict.copy， 并修改source.dict
  get_source_dict_copy(source_dict_label, source_dict, source_dict_copy)