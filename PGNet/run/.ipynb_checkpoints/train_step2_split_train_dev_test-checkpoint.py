import random
from collections import Counter
import os
import re
import sys
import jieba
import pickle
# import discriminator
# import data


# 525148 {420118, 52515, 52515}, 98348 {78678, 9835, 9835}
# 524251, 98057

class BlackWords(object):
  def __init__(self, blackWord_file):
    with open(blackWord_file, 'r', encoding='utf-8') as f:
      self.black_list = [_.strip() for _ in f.readlines()]

  def getWords(self):
    return self.black_list

def add_period(text, all_word_end_period):
    split = [_.strip() for _ in re.split(r'[。，；？！,;?!]', text) if _.strip() != '']
    buf_end_word_freq = list()
    for index, tmp in enumerate(split):
        last = tmp[-1]
        if last in all_word_end_period:
            buf_end_word_freq.append(all_word_end_period[last])
    if len(buf_end_word_freq) < 1:
        return ""
    max_count = -1
    max_index = 0
    for index in range(1, len(buf_end_word_freq)-2):
        if buf_end_word_freq[index] >= max_count:
            max_count = buf_end_word_freq[index]
            max_index = index
    length = 0
    for i in range(max_index+1):
        length += len(split[i]) + 1
    length -= 1
    list_text = list(text)
    list_text[length] = '。'
    final = ''.join(list_text)
    return final


def split_train_dev_test(file, out_data_path, blackWord_path):
  pattern = re.compile(r'[^\u4e00-\u9fa5]+')
  pairs = {}
  black_list = BlackWords(blackWord_path)
  filter_reasons = {}
  filter_reasons['len(kb) < 10'] = 0
  filter_reasons['len(text) < 55 or len(text) > 90'] = 0
  filter_reasons['not is_valid_sku_title(skuTitle, black_list)'] = 0
  filter_reasons['not is_valid_sku_title(text, black_list)'] = 0
  filter_reasons['not is_valid_daren_title(title)'] = 0
  filter_reasons['text or kbStr is null'] = 0
  filter_reasons['noperiod'] = 0
  filter_reasons['subsent_len'] = 0
  badSubSents = {}
  all_word_end_period = pickle.load(open('/home/liangjiahui8/notespace/shouji/all_word_end_period', 'rb'))
  with open(file, 'r', encoding='utf-8') as f:
    id = 0
    text = ""
    kbStr = ""
    skuTitle = ""
    kb = {}
    while 1:
      line = f.readline()
      if not line:
        print(id)
        break

      line = line.strip('\n')
      if line.startswith('AlphaWriting'):
        id = int(line.split('-')[1])
      if line.startswith('Title:'):
        title = line[7:]
      if line.startswith('Text:'):
        text = re.sub(r'\s+', ' ', line[6:])
      if line.startswith('Sku:'):
        sku = line[5:]
      if line.startswith('KB:'):
        kv = line[4:].split(' ||| ')
        key = kv[0].strip()
        value = kv[1].strip()
        if key == 'SkuName':
          skuTitle = value
        if (len(kv) == 2 and key != "" and value != "" and value != "--" and key != "Lables" and key != "Labels"):
          kbStr = kbStr + key + ' $$ ' + value + ' ## '
          if key not in kb:
            kb[key] = []
          kb[key].append(value)
      if line == "":
        if text != "" and kbStr != "":
          kbStr = kbStr[:-4].strip()
          
          if (len(kb) < 10):
            filter_reasons['len(kb) < 10'] += 1
            kbStr =  ""
            continue
          
          if len(text) < 55 or len(text) > 95:
            filter_reasons['len(text) < 55 or len(text) > 90'] += 1
            kbStr =  ""
            continue
          skuTitle = re.sub(r'[【】]', '', skuTitle)
#           if not is_valid_sku_title(skuTitle, black_list):
#             filter_reasons['not is_valid_sku_title(skuTitle, black_list)'] += 1
#             kbStr =  ""
#             continue
                    
          if not is_valid_sku_title(text, black_list):
            filter_reasons['not is_valid_sku_title(text, black_list)'] += 1
            kbStr =  ""
            continue
            
          text = text.replace('；','。').replace('！','。').replace(';','。').replace('!','。').replace(',','，')
          if not text[-1] == '。':
            ls = list(text)
            ls.append('。')
            text = ''.join(ls)
          tmp_list = [_.strip() for _ in re.split(r'[。，；？！,;?!]', text) if _.strip() != '']
          if len([i for i in text.strip()[:-1] if i=='。']) < 1:
            if len(tmp_list) < 5:
                filter_reasons['noperiod'] += 1
                kbStr =  ""
                continue
            else:
                text = add_period(text, all_word_end_period)
                if len(text) < 1:
                    filter_reasons['noperiod'] += 1
                    kbStr =  ""
                    continue
          tmp_list = [_.strip() for _ in re.split(r'[。，；？！,;?!]', text) if _.strip() != '']
          F = True
          for i in tmp_list:
            chineseText = re.sub(pattern, '', i)
            if len(chineseText) > 20 or len(chineseText) < 3:
              F = False
              break
          if F==False:
            kbStr =  ""
            filter_reasons['subsent_len'] += 1
            continue 
          
          for x in tmp_list:
            if i not in badSubSents:
              badSubSents[i] = 0
            badSubSents[i] += 1
              
          if sku not in pairs:
            pairs[sku] = []
          pairs[sku].append((id, kbStr, text, title))
        else:
          filter_reasons['text or kbStr is null'] += 1
          print(id)
        kbStr = ""
        kb = {}
        text = ""
        skuTitle = ""
  print('=============filter_reasons==============')
  print(filter_reasons)
  print('=============filter_reasons==============')

  with open(file + '.badcase', 'w', encoding='utf-8') as w:
    for k, v in sorted(badSubSents.items(), key=lambda x: x[1], reverse=True):
      w.write(k + '\t' + str(v) + '\n')
    
  pairs = list(pairs.items())

  random.shuffle(pairs)
  # train_num = round(len(pairs) * 0.8)
  # dev_num = round(len(pairs) * 0.1)
  train_num = len(pairs) - 4000
  dev_num = 2000
  train = pairs[:train_num]
  dev = pairs[train_num:train_num + dev_num]
  test = pairs[train_num + dev_num:]
  print("======%d{%d,%d,%d}=====" % (len(pairs), train_num, dev_num, len(pairs) - train_num - dev_num))

  write_train_file(out_data_path, train, "train")
  write_dev_test_file(out_data_path, dev, "dev")
  write_dev_test_file(out_data_path, test, "test")
  write_total_file(out_data_path, train, dev, "total")

def write_fake_writing(file, dataset, tag):
  with open(os.path.join(file, tag), 'w', encoding='utf-8') as f_table:
    len_  = len(dataset)
    for i in range(len_):
      f_table.write("\n")

def write_train_file(file, dataset, tag):
  with open(os.path.join(file, tag+ ".table"), 'w', encoding='utf-8') as f_table:
    with open(os.path.join(file, tag + ".text"), 'w', encoding='utf-8') as f_text:
      with open(os.path.join(file, tag + ".title"), 'w', encoding='utf-8') as f_title:
        with open(os.path.join(file, tag + ".id"), 'w', encoding='utf-8') as f_id:
          with open(os.path.join(file, tag + ".sku"), 'w', encoding='utf-8') as f_sku:
            with open(os.path.join(file, "train.fakeConstraint"), 'w', encoding='utf-8') as f_fake_ocr:
              with open(os.path.join(file, "train.fakeValidOcr"), 'w', encoding='utf-8') as f_fake:
                for sku, tuples in dataset:
                  for tuple in tuples:
                    id, table, text, title = tuple
                    f_table.write(table + "\n")
                    f_text.write(text + "\n")
                    f_title.write(title + "\n")
                    f_id.write(str(id) + "\n")
                    f_sku.write(sku + "\n")
                    f_fake_ocr.write('\n')
                    f_fake.write('\n')


def write_dev_test_file(file, dataset, tag):
  cnt = 0
  with open(os.path.join(file, tag + ".table"), 'w', encoding='utf-8') as f_table:
    with open(os.path.join(file, tag + ".sku"), 'w', encoding='utf-8') as f_sku:
      with open(os.path.join(file, tag + ".id"), 'w', encoding='utf-8') as f_id:
        with open(os.path.join(file, tag + ".title"), 'w', encoding='utf-8') as f_title:
          with open(os.path.join(file, tag + ".text"), 'w', encoding='utf-8') as f_text:
            with open(os.path.join(file, tag + ".id.multi"), 'w', encoding='utf-8') as f_id_multi:
              with open(os.path.join(file, tag + ".title.multi"), 'w', encoding='utf-8') as f_title_multi:
                with open(os.path.join(file, tag + ".text.multi"), 'w', encoding='utf-8') as f_text_multi:
                  with open(os.path.join(file, tag + ".number.multi"), 'w', encoding='utf-8') as f_number:
                    with open(os.path.join(file, "dev.fakewriting"), 'w', encoding='utf-8') as f_fakew:
                      with open(os.path.join(file, "dev.validSku.flag"), 'w', encoding='utf-8') as f_flag:
                        for sku, tuples in dataset:
                          id, table, text, title = tuples[0]
                          f_table.write(table + "\n")
                          f_id.write(str(id) + "\n")
                          f_title.write(title + "\n")
                          f_text.write(' '.join(jieba.lcut(text)) + "\n")
                          f_sku.write(sku + "\n")
                          cnt += len(tuples)
                          f_number.write(str(len(tuples)) + "\n")
                          f_flag.write('1'+'\n')
                          f_fakew.write('\n')
                          for tuple in tuples:
                            id, table, text, title = tuple
                            f_id_multi.write(str(id) + "\n")
                            f_title_multi.write(title + "\n")
                            f_text_multi.write(text + "\n")
  print(cnt)


def write_total_file(file, dataset1, dataset2,  tag):
  cnt = 0
  tmp_sku={}
  with open(os.path.join(file, tag + ".sku"), 'w', encoding='utf-8') as f_sku:
    for sku, tuples in dataset1:
      if sku not in tmp_sku:
        f_sku.write(sku + "\n")
        tmp_sku[sku]=1
      for sku, tuples in dataset2:
        if sku not in tmp_sku:
          f_sku.write(sku + "\n")
          tmp_sku[sku]=1
          
def load_kb(sku_file, kb_file):
  with open(sku_file, 'r', encoding='utf-8') as sr:
    sku_list = [sku.strip() for sku in sr.readlines()]
  kb = {}
  sku2title = {}
  with open(kb_file, "r", encoding='utf-8') as f:
    index = 0
    for line in f:
      sku = sku_list[index]
      if sku in kb:
        index += 1
        continue
      kb[sku] = {}
      sku2title[sku] = ''
      kvs = line.strip().lower().split(' ## ')
      for kv in kvs:
        parts = kv.split(' $$ ')
        key = parts[0].replace(' ', '').replace('\t', '').strip()
        value = parts[1].replace(' ', '').replace('\t', '').strip()

        if (key.lower() != 'skuname'):
          if key not in kb[sku]:
            kb[sku][key] = []
          kb[sku][key].append(value)
        else:
          sku2title[sku] = value
      index += 1
  return kb, sku2title


def load_index2Info(kb_file, title_file):
  # index = 0
  # index2sku = {}
  # with open(sku_file, 'r', encoding='utf-8') as f:
  #   while (True):
  #     line = f.readline()
  #     if (not line):
  #       break
  #     line = line.strip()
  #     index2sku[index] = line
  #     index += 1

  index = 0
  index2kb = {}
  with open(kb_file, 'r', encoding='utf-8') as f:
    while (True):
      line = f.readline()
      if (not line):
        break
      line = line.strip()
      index2kb[index] = line
      index += 1

  index = 0
  index2darenTitle = {}
  with open(title_file, 'r', encoding='utf-8') as f:
    while (True):
      line = f.readline()
      if (not line):
        break
      line = line.strip()
      index2darenTitle[index] = line
      index += 1

  # index = 0
  # index2writing = {}
  # with open(writing_file, 'r', encoding='utf-8') as f:
  #   while(True):
  #     line = f.readline()
  #     if(not line):
  #       break
  #     line = line.strip()
  #     writing = line.replace(' ', '').replace('\t', '')
  #     charCount = Counter(writing)
  #     if charCount['"'] % 2 != 0:
  #       writing = writing.replace('"', '')
  #     if charCount['\''] % 2 != 0:
  #       writing = writing.replace('\'', '')
  #     index2writing[index] = line
  # return index2sku, index2kb, index2darenTitle, index2writing
  return index2kb, index2darenTitle


def is_valid_daren_title(title):
  pat = r'[\u4e00-\u9fa5]'
  tmp = re.sub(pattern=pat, repl='aa', string=title.replace(' ', ''))
  if (len(tmp) < 12 or len(tmp) > 20):
    return False
  # if (len(title.split(' ')) != 3):
  #   return False

  return True


def is_valid_sku_title(text, black_list):
  for w in black_list.getWords():
    if w in text:
      return False
  return True


def training_data_filter(root, domain):
  # prepare: run 'review-trainingData' in 'discriminator.py', and get the file 'train.review.log'

  sku_file = os.path.join(root, r'chuju_rawData/train.sku')
  kb_file = os.path.join(root, r'chuju_rawData/train.table')
  title_file = os.path.join(root, r'chuju_rawData/train.title')
  # writing_file = os.path.join(root, r'train.text')
  review_log_File = os.path.join(root, r'chuju_rawData/train.review.log')
  bad_sku_File = os.path.join(root, r'chuju_rawData/train.sku.bad')

  kb, sku2title = load_kb(sku_file, kb_file)
  index2kb, index2darenTitle = load_index2Info(kb_file, title_file)
  blackWord_path = os.path.join(root, r'blackList.txt')  # r'D:\Research\Projects\AlphaWriting\data\blackList.txt'
  black_list = BlackWords(blackWord_path)

  print('finish loading data')
  print(len(index2kb), len(index2darenTitle))

  index = 0
  sw_sku = open(os.path.join(root, 'filtered_data', 'train.sku'), 'w', encoding='utf-8')
  sw_text = open(os.path.join(root, 'filtered_data', 'train.text'), 'w', encoding='utf-8')
  sw_title = open(os.path.join(root, 'filtered_data', 'train.title'), 'w', encoding='utf-8')
  sw_table = open(os.path.join(root, 'filtered_data', 'train.table'), 'w', encoding='utf-8')
  sw_cons = open(os.path.join(root, 'filtered_data', 'train.fakeConstraint'), 'w', encoding='utf-8')

  cnt = 0
  fewKVCnt = 0
  notPassCnt = 0
  puncCnt = 0
  titleCnt = 0
  skuTitleCnt = 0
  repeatCnt = 0
  unique = set()
  with open(review_log_File, 'r', encoding='utf-8') as f:
    f.readline()
    while (True):
      line = f.readline()
      if (not line):
        break
      if index >= 504818:
        break
      line = line.strip()
      parts = line.split('\t')
      label = parts[1]
      sku = parts[2]
      cat = parts[4]
      writing = parts[5]
      if writing.endswith('，') or writing.endswith('；'):
        writing = writing[:-1] + '。'
      if not any([writing.endswith(punc) for punc in ['。', '！', '？']]):
        writing = writing + '。'
      tokenized_writing = parts[6]
      lengthError = parts[7]
      shortSentLengthError = parts[9]
      punctuationError = parts[10]
      badWordError = parts[11]
      notConsistentError = parts[13]
      darenTitle = index2darenTitle[index]
      table = index2kb[index]
      kvs = kb[sku]
      skuTitle = sku2title[sku]

      if (len(kvs) < 11):  # 6 + 5
        fewKVCnt += 1
        index += 1
        continue
      if label == 'NotPass!':
        notPassCnt += 1
        index += 1
        continue
      # if punctuationError == 'True':
      #   puncCnt += 1
      #   index += 1
      #   continue
      if not is_valid_daren_title(darenTitle):
        titleCnt += 1
        index += 1
        continue
      if not is_valid_sku_title(skuTitle, black_list):
        skuTitleCnt += 1
        index += 1
        continue
      if sku + '\t' + table + '\t' + writing + '\t' + darenTitle in unique:
        repeatCnt += 1
        index += 1
        continue

      sw_sku.write(sku + '\n')
      sw_table.write(table + '\n')
      sw_title.write(darenTitle + '\n')
      sw_text.write(writing + '\n')
      sw_cons.write('\n')
      unique.add(sku + '\t' + table + '\t' + writing + '\t' + darenTitle)
      index += 1
      cnt += 1

  sw_sku.close()
  sw_table.close()
  sw_title.close()
  sw_text.close()
  sw_cons.close()

  print(cnt)
  print(fewKVCnt)
  print(notPassCnt)
  print(puncCnt)
  print(titleCnt)
  print(skuTitleCnt)
  print(repeatCnt)


def test_dev_data_filter(root, domain, dataset, thirdCats):
  # prepare: run 'review-trainingData' in 'discriminator.py', and get the file 'train.review.log'
  # root = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi'
  sku_file = os.path.join(root, dataset + '.sku')
  kb_file = os.path.join(root, dataset + '.table')
  title_file = os.path.join(root, dataset + '.title')
  # writing_file = os.path.join(root, r'train.text')
  review_log_File = os.path.join(root, dataset + '.review.log')
  bad_sku_File = os.path.join(root, dataset + '.sku.bad')

  kb, sku2title = load_kb(sku_file, kb_file)
  index2kb, index2darenTitle = load_index2Info(kb_file, title_file)
  blackWord_path = r'D:\Research\Projects\AlphaWriting\data\blackList.txt'
  black_list = data.BlackWords(blackWord_path)

  print('finish loading data')
  print(len(index2kb), len(index2darenTitle))

  index = 0
  sw_sku = open(os.path.join(root, 'filtered_data', dataset + '.sku'), 'w', encoding='utf-8')
  sw_text = open(os.path.join(root, 'filtered_data', dataset + '.text'), 'w', encoding='utf-8')
  sw_title = open(os.path.join(root, 'filtered_data', dataset + '.title'), 'w', encoding='utf-8')
  sw_table = open(os.path.join(root, 'filtered_data', dataset + '.table'), 'w', encoding='utf-8')
  sw_cons = open(os.path.join(root, 'filtered_data', dataset + '.fakeConstraint'), 'w', encoding='utf-8')
  sw_flag = open(os.path.join(root, 'filtered_data', dataset + '.flag'), 'w', encoding='utf-8')

  cnt = 0
  catCnt = 0
  fewKVCnt = 0
  notPassCnt = 0
  puncCnt = 0
  titleCnt = 0
  skuTitleCnt = 0
  repeatCnt = 0
  unique = set()
  with open(review_log_File, 'r', encoding='utf-8') as f:
    f.readline()
    while (True):
      line = f.readline()
      if (not line):
        break
      if index >= 2000:
        break
      line = line.strip()
      parts = line.split('\t')
      label = parts[1]
      sku = parts[2]
      cat = parts[4]
      writing = parts[5]
      if writing.endswith('，') or writing.endswith('；'):
        writing = writing[:-1] + '。'
      if not any([writing.endswith(punc) for punc in ['。', '！', '？']]):
        writing = writing + '。'
      tokenized_writing = parts[6]
      lengthError = parts[7]
      shortSentLengthError = parts[9]
      punctuationError = parts[10]
      badWordError = parts[11]
      notConsistentError = parts[13]
      darenTitle = index2darenTitle[index]
      table = index2kb[index]
      kvs = kb[sku]
      skuTitle = sku2title[sku]

      if cat not in thirdCats:
        catCnt += 1
        index += 1
        continue
      if (len(kvs) < 11):  # 6 + 5
        fewKVCnt += 1
        index += 1
        continue
      if label == 'NotPass!':
        notPassCnt += 1
        index += 1
        continue
      # if punctuationError == 'True':
      #   puncCnt += 1
      #   index += 1
      #   continue
      if not is_valid_daren_title(darenTitle):
        titleCnt += 1
        index += 1
        continue
      if not is_valid_sku_title(skuTitle, black_list):
        skuTitleCnt += 1
        index += 1
        continue
      if sku + '\t' + table + '\t' + writing + '\t' + darenTitle in unique:
        repeatCnt += 1
        index += 1
        continue

      sw_sku.write(sku + '\n')
      sw_table.write(table + '\n')
      sw_title.write(darenTitle + '\n')
      sw_text.write(writing + '\n')
      sw_cons.write('\n')
      sw_flag.write('1\n')

      unique.add(sku + '\t' + table + '\t' + writing + '\t' + darenTitle)
      index += 1
      cnt += 1

  sw_sku.close()
  sw_table.close()
  sw_title.close()
  sw_text.close()
  sw_cons.close()
  sw_flag.close()

  print(cnt)
  print(catCnt)
  print(fewKVCnt)
  print(notPassCnt)
  print(puncCnt)
  print(titleCnt)
  print(skuTitleCnt)
  print(repeatCnt)

def write_k_file(path1, path2):
 r1 = open(path2, 'r')
 w1 = open(path1+'/'+'k.txt','w')
 dic = {}
 for line in r1.readlines():
   line1 = line.strip().split(' ## ')
   for kv in line1:
      k,v = kv.split(' $$ ')
      k = k.strip()
      v = v.strip()
      if k not in dic:
         dic[k]=1
 w1.write('sku_name'+'\n')
 w1.write('brandname_full'+'\n')
 w1.write('brandname_cn'+'\n')
 w1.write('brandname_en'+'\n')
 w1.write('cate1_cd'+'\n')
 w1.write('cate2_cd'+'\n')
 w1.write('cate3_cd'+'\n')
 for k in dic:
   w1.write(k+'\n') 

if __name__ == "__main__":
  #root = r'/export/homes/baojunwei/alpha-w-pipeline/data'
  root = os.path.join(sys.argv[1], 'data')
  domain = sys.argv[2]
  folder = sys.argv[3]
  kb_writing_file = os.path.join(root, domain + '_' + folder, 'train' + '.writing.reduced.joinKB')
  pred_file = os.path.join(root, domain + '_' + folder+"_latest")
  blackWord_path = os.path.join(root, 'blackList.txt')  # r'D:\Research\Projects\AlphaWriting\data\blackList.txt'
  out_data_path = os.path.join(root, domain)
  split_train_dev_test(kb_writing_file, out_data_path, blackWord_path)
  write_k_file(pred_file, os.path.join(root, domain, "train.table"))
