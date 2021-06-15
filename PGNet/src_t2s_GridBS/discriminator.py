import tensorflow as tf
import numpy as np
import data
import collections

class ConsistentDiscriminator(object):
  def __init__(self, ckg, sku_file, kb_file):
    self.kb = self.load_kb(sku_file, kb_file)
    self.ckg = ckg

  def load_kb(self, sku_file, kb_file):
    with open(sku_file, 'r', encoding='utf-8') as sr:
      sku_list = [sku.strip() for sku in sr.readlines()]
    kb = {}
    with open(kb_file, "r", encoding='utf-8') as f:
      index = 0
      for line in f:
        sku = sku_list[index]
        kb[sku]={}
        kvs = line.strip().lower().split(' ## ')
        for kv in kvs:
          parts = kv.split(' $$ ')
          key = parts[0].replace(' ', '').replace('\t', '').strip()
          value = parts[1].replace(' ', '').replace('\t', '').strip()

          if (key.lower() != 'skuname'):
            if key not in kb[sku]:
              kb[sku][key] = []
            kb[sku][key].append(value)
        index += 1
    return kb

  def discriminate(self, sku, writing):
    sku_kb = self.kb.get(sku, None)
    if(not sku_kb):
      print("ERROR: not contain sku: %s"%sku)
      return True
    writing = writing.replace(' ', '').replace('\t', '')
    return self.discriminate_withKB(sku_kb, writing)

  def discriminate_withKB(self, sku_kb, writing):
    blackAttrList = ['skuname', 'brandname_full', 'brandname_en', 'labels'
                     'CategoryL1'.lower(), 'CategoryL2'.lower(), 'CategoryL3'.lower(),
                     '特色推荐', '特色功能', '产品特点', '产品特色',
                     '产品类型', '规格_水温调节范围', '主体_品牌']
    # blackAttrList = ['brandname_full', 'brandname_en', 'brandname_cn']

    blackValueList = ['有', '无', '是', '否', '支持', '不支持', '其他', '其它']

    cat3 = sku_kb['CategoryL3'.lower()][0]
    # print('sku third cat %s' %cat3)
    cat_ckg = self.ckg.get_cat_ckg(cat3)
    for catAttr, catValues in cat_ckg.items():
      if catAttr in sku_kb and catAttr not in blackAttrList:
        intersect = []
        for catValueSet in catValues:
          if any([catValue in blackValueList for catValue in catValueSet]):
            continue
          for v in sku_kb[catAttr]:
            if v in catValueSet:
              intersect.append(v)
          # intersect = set(sku_kb[catAttr]).intersection(set(catValueSet))
        if len(intersect) == 0:
          continue
        for catValueSet in catValues:
          for catValue in catValueSet:
            # if any([catValue in catValueSet and catValue not in intersect and catValue in writing for catValue in catValueSet])
            if len(list(set(catValueSet).intersection(intersect))) == 0 and catValue in writing and '免去' not in writing:
              sku_value = " | ".join(sku_kb[catAttr])
              if(any([catValue in sku_v or sku_v in catValue for sku_v in intersect])):
                continue
              # for k, v in sku_kb.items():
              #   print("%s ||| %s" % (k, ' | '.join(v)))
              print('Not consistent: KB uses [%s ||| %s] but writing contains [%s]'
                    % (catAttr, sku_value, catValue))
              return False
    return True

def judge_one(discriminator, black_list, sku, writing, index):
  writing = writing.replace(' ','')
  print("Index-%d:==================="%(index+1))
  flag = True
  errorTypes = []
  l = len(writing)
  if(l < 50 or l > 90):
    print('Not Passed %d (too short or long %d): [SKU:%s] [Writing:%s]' % (index + 1, l, sku, writing))
    flag = False
    errorTypes.append('LengthError')

  for w in black_list.getWords():
    if w in writing:
      print('Not Passed %d (containing bad word %s): [SKU:%s] [Writing:%s]' % (index + 1, w, sku, writing))
      flag = False
      errorTypes.append('BadWordError')

  consistent = discriminator.discriminate(sku, writing)
  if (not consistent):
    print('Not Passed %d (KB-Writing Not Consistent): [SKU:%s] [Writing:%s]' % (index + 1, sku, writing))
    flag = False
    errorTypes.append('NotConsistentError')

  if flag:
    print("Passing!")
  print('')
  return errorTypes

def judge_file(sku_file, predictin_file, discriminator, black_list):
  with open(sku_file, 'r', encoding='utf-8') as sr:
    sku_list = [sku.strip() for sku in sr.readlines()]

  errorCounter = collections.Counter()
  correctNum = 0
  with open(predictin_file, 'r', encoding='utf-8') as f:
    index = 0
    while(True):
      line = f.readline()
      if(not line):
        break
      writing = line.strip()
      sku = sku_list[index]
      errorTypes = judge_one(discriminator, black_list, sku, writing, index)
      index += 1
      errorCounter.update(errorTypes)
      if(len(errorTypes) == 0):
        correctNum += 1

  print('Passed Number %d/%d' %(correctNum, index))
  print('Not Passed Number %d/%d' %(index - correctNum, index))
  print(errorCounter.most_common())
  

if __name__ == '__main__':  
  #ckg_path = r'/home/baojunwei/project/alpha_writing/analysis/groupByCat3_withValue/*'
  ckg_path = r'/home/baojunwei/project/alpha_writing/data/ckg/*'
  blackWord_path = r'/home/baojunwei/project/alpha_writing/data/black_list/blackList.txt'
  ckg = data.CKG(ckg_path)  
  black_list = data.BlackWords(blackWord_path)
  
  sku_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi/test.sku'
  kb_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi/test.table'
  discriminator = ConsistentDiscriminator(ckg, sku_file, kb_file)

  predictin_file = r'/home/baojunwei/project/alpha_writing/log/jiayongdianqi/kb2writing_t2s_bigVocab/decode_test_20maxenc_4beam_20mindec_50maxdec_force_decoding_ckpt-561730/decoded/decoded.txt'
  #judge_file(sku_file, predictin_file, discriminator, black_list)
  print('===============================')

  predictin_file = r'/home/baojunwei/project/alpha_writing/log/jiayongdianqi/kb2writing_t2s_bigVocab/decode_test_20maxenc_4beam_20mindec_50maxdec_force_decoding_ckg_ckpt-561730/decoded/decoded.txt'
  #judge_file(sku_file, predictin_file, discriminator, black_list)
  print('===============================')

  predictin_file = r'/home/baojunwei/project/alpha_writing/log/jiayongdianqi/kb2writing_t2s_bigVocab/decode_test_20maxenc_10beam_20mindec_50maxdec_force_decoding_ckg_ckpt-561730/decoded/decoded.txt'
  #judge_file(sku_file, predictin_file, discriminator, black_list)
  print('===============================')

  

  sku_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin/xuanpin.merge.sku'
  kb_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin/xuanpin.merge.sku.table'
  discriminator = ConsistentDiscriminator(ckg, sku_file, kb_file)
  
  predictin_file = r'/home/baojunwei/project/alpha_writing/log/jiayongdianqi/kb2writing_t2s_bigVocab/decode_xuanpin_20maxenc_4beam_20mindec_50maxdec_force_decoding_ckpt-561730/decoded/decoded.txt'
  judge_file(sku_file, predictin_file, discriminator, black_list)
  print('===============================')

  predictin_file = r'/home/baojunwei/project/alpha_writing/log/jiayongdianqi/kb2writing_t2s_bigVocab/decode_xuanpin_20maxenc_4beam_20mindec_50maxdec_force_decoding_ckg_ckpt-561730/decoded/decoded.txt'
  judge_file(sku_file, predictin_file, discriminator, black_list)
  print('===============================')

  predictin_file = r'/home/baojunwei/project/alpha_writing/log/jiayongdianqi/kb2writing_t2s_bigVocab/decode_xuanpin_20maxenc_10beam_20mindec_50maxdec_force_decoding_ckg_ckpt-561730/decoded/decoded.txt'
  judge_file(sku_file, predictin_file, discriminator, black_list)
  print('===============================')



