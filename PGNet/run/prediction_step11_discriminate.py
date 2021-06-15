import tensorflow as tf
import numpy as np
#import data
import collections
import re
from collections import Counter
import jieba
import os
import math
import glob
import sys
class BlackWords(object):
  def __init__(self, blackWord_file):
    with open(blackWord_file, 'r', encoding='utf-8') as f:
      self.black_list = [_.strip() for _ in f.readlines()]
  def getWords(self):
    return self.black_list


class CKG(object):
  def __init__(self, ckg_file_path):
    import re
    self.ckg = {}
    filelist = glob.glob(ckg_file_path) # get the list of datafiles
    for ckg_file in filelist:
      cat = ckg_file.split('.')[-3].lower().replace(' ','')
      print('Loading CKG for category %s' %cat)
      if cat not in self.ckg:
        self.ckg[cat] = {}
      with open(ckg_file, 'r', encoding='utf-8') as f:
        attr = ''
        values = []
        while(True):
          line = f.readline()
          if(not line):
            break
          line = line.strip('\n').lower()
          if(not line == '' and not line.startswith('\t')):
            attrPatrs = line.strip().split('\t:\t')
            if int(attrPatrs[1]) < 10:
              continue
            attr = attrPatrs[0]
          elif(line.startswith('\t')):
            valueParts = line.strip().split(':\t')
            # if int(valueParts[1]) < 2:
            #   continue
            value = valueParts[0].replace(' ', '').replace('\t', '').split(r'/')
            values.append(value)
          elif(line == ''):
            if attr.lower() == 'BrandName_cn'.lower():
              values = [[_ for _ in v if not re.match('[a-zA-Z]', _)] for v in values]
              values = [v for v in values if len(v) != 0]
            if attr != '' and len(values) > 1:
              self.ckg[cat][attr]=values
            attr = ''
            values = []

  def get_cat_ckg(self, cat):
    if cat in self.ckg:
      return self.ckg[cat]
    else:
      # print('there is no ckg for %s category' %cat)
      return {}


class ConsistentDiscriminator(object):
  def __init__(self, ckg, sku_file, kb_file):
    self.kb, self.sku2title = self.load_kb(sku_file, kb_file)
    self.ckg = ckg

  def load_kb(self, sku_file, kb_file):
    with open(sku_file, 'r', encoding='utf-8') as sr:
      sku_list = [sku.strip() for sku in sr.readlines()]
    kb = {}
    sku2title = {}
    with open(kb_file, "r", encoding='utf-8') as f:
      index = 0
      for line in f:
        sku = sku_list[index]
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

  def discriminate(self, sku, writing):
    sku_kb = self.kb.get(sku, None)
    if (not sku_kb):
      print("ERROR: not contain sku: %s" % sku)
      return True
    writing = writing.replace(' ', '').replace('\t', '')
    return self.discriminate_withKB(sku_kb, writing)

  def discriminate_withKB(self, sku_kb, writing):
    blackAttrList = ['skuname', 'brandname_full', 'brandname_en', 'labels',
                     'CategoryL1'.lower(), 'CategoryL2'.lower(),
                     'CategoryL3'.lower(),
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
            if len(list(set(catValueSet).intersection(intersect))) == 0 and \
                catValue in writing and \
                '免去' not in writing and \
                '避免' not in writing and \
                '免除' not in writing and \
                '告别' not in writing and \
                '解除' not in writing and \
                '省去' not in writing and \
                '无需' not in writing and \
                '不用' not in writing:
              sku_value = " | ".join(sku_kb[catAttr])
              if (any([catValue in sku_v or sku_v in catValue for sku_v in intersect])):
                continue
              if (any([v in writing for v in sku_kb[catAttr]])):
                continue
              # for k, v in sku_kb.items():
              #   print("%s ||| %s" % (k, ' | '.join(v)))
              # print('Not consistent: KB uses [%s ||| %s] but writing contains [%s]'
              #       % (catAttr, sku_value, catValue))
              reason = 'KB: [{0} ||| {1}] writing [{2}]'.format(catAttr, sku_value, catValue)
              return False, reason
    return True, 'Good!'


def discriminate_length(writing):
  l = len(writing)

  if (l < 50):
    # print('Not Passed %d (too short or long %d): [SKU:%s] [Writing:%s]' % (index + 1, l, sku, writing))
    return False, l
  elif (l > 90):
    # print('Not Passed %d (too short or long %d): [SKU:%s] [Writing:%s]' % (index + 1, l, sku, writing))
    return False, l

  return True, l


def discriminate_badWord(writing, black_list):
  badWrods = []
  for w in black_list.getWords():
    if w in writing:
      badWrods.append(w)
  if (len(badWrods) > 0):
    return False, badWrods
  return True, badWrods


def discriminate_allComma(writing):
  sentences = re.split('[；。！]', writing.strip('。；！'))
  sents_len = len(sentences)
  if (sents_len <= 1):
    return False
  return True


def discriminate_shortSentLen(writing):
  sentences = re.split('[；。！]', writing.strip('。；！'))
  sents_len = len(sentences)
  allSentences = re.split('[，。；！]', writing.strip('。；！'))
  allSentsLen = len(allSentences)
  # if (sents_len <= 1 or allSentsLen >= 16 or allSentsLen <= 2):
  if (allSentsLen >= 16 or allSentsLen <= 2):
    return False
  return True


def judge_one(discriminator, black_list, sku, writing, index, sw):
  writing = writing.replace(' ', '').replace('\t', '')
  charCount = Counter(writing)
  if charCount['"'] % 2 != 0:
    writing = writing.replace('"', '')
  if charCount['\''] % 2 != 0:
    writing = writing.replace('\'', '')

  tokenized_writing = ' '.join(jieba.cut(writing))

  cat = discriminator.kb[sku]['CategoryL3'.lower()][0]

  result = 'Pass!'
  # print("Index-%d=============="%(index+1))
  errorFlag = False
  errorTypes = []
  # l = len(writing)

  lengthErrorFlag = False
  f, l = discriminate_length(writing)
  if (not f):
    lengthErrorFlag = True
    errorFlag = True
    errorTypes.append('LengthError')

  shortSentLength = False
  if (not discriminate_shortSentLen(writing)):
    shortSentLength = True
    errorFlag = True
    errorTypes.append('ShortSentLengthError')

  punctuationErrorFlag = False
  if (not discriminate_allComma(writing)):
    punctuationErrorFlag = True
    # errorFlag = True
    errorTypes.append('PunctuationError')

  badWordFlag = False
  badWrods = []
  f, badWrods = discriminate_badWord(writing, black_list)
  if (not f):
    badWordFlag = True
    errorFlag = True
    errorTypes.append('BadWordError')

  # for w in black_list.getWords():
  #   if w in writing:
  #     # print('Not Passed %d (containing bad word %s): [SKU:%s] [Writing:%s]' % (index + 1, w, sku, writing))
  #     badWordFlag = True
  #     badWrods.append(w)
  #     errorFlag = True
  #     if(f):
  #       errorTypes.append('BadWordError')
  #       f = False

  consistentFlag, resaon = discriminator.discriminate(sku, writing)
  if (not consistentFlag):
    # print('Not Passed %d (KB-Writing Not Consistent): [SKU:%s] [Writing:%s]' % (index + 1, sku, writing))
    errorFlag = True
    errorTypes.append('NotConsistentError')

  if errorFlag:
    result = 'NotPass!'
  sku_url = 'https://item.jd.com/' + sku + '.html'
  logStr = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}" \
    .format(index + 1, result, sku, sku_url, cat, writing, tokenized_writing, lengthErrorFlag, l, shortSentLength, punctuationErrorFlag, badWordFlag,
            ' || '.join(badWrods), not consistentFlag, resaon)
  sw.write(logStr + '\n')
  # print(logStr)
  return errorTypes, logStr


def judge_file(sku_file, predictin_file, discriminator, black_list, logFile, bad_sku_File):
  sku2res = {}
  with open(logFile, 'w', encoding='utf-8') as sw:
    with open(bad_sku_File, 'w', encoding='utf-8') as sw1:
      logStr = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}". \
        format('index',
               'reviewResult',
               'sku',
               'sku_url',
               'Category',
               'writing',
               'tokenized_writing',
               'lengthError',
               'writingLength',
               'shortSentLengthError',
               'punctuationError',
               'badWordError',
               'badWrodReason',
               'notConsistentError',
               'notConsistentResaon')
      sw.write(logStr + '\n')
      # print(logStr)
      with open(sku_file, 'r', encoding='utf-8') as sr:
        sku_list = [sku.strip() for sku in sr.readlines()]
      bad_sku_list = []
      errorCounter = Counter()
      correctNum = 0
      with open(predictin_file, 'r', encoding='utf-8') as f:
        index = 0
        while (True):
          line = f.readline()
          if (not line):
            break
          writing = line.strip()
          sku = sku_list[index]
          errorTypes, res = judge_one(discriminator, black_list, sku, writing, index, sw)
          sku2res[sku] = res
          index += 1
          errorCounter.update(errorTypes)
          if(len(errorTypes) == 0):
            correctNum += 1
          elif(len(errorTypes) == 1 and errorTypes[0] == 'PunctuationError'):
            correctNum += 1
          else:
            bad_sku_list.append(sku)

      sw.write('Passed Number %d/%d\n' % (correctNum, index))
      sw.write('Not Passed Number %d/%d\n' % (index - correctNum, index))
      sw.write(str(errorCounter.most_common()) + '\n')

      for sku in bad_sku_list:
        sw1.write(sku + '\n')
  return sku2res


def print_discriminater_res_for_specific_skus(skus, sku2res):
  for sku in skus:
    if sku in sku2res:
      parts = sku2res[sku].split('\t')
      result = parts[1]
      writing = parts[5]
      length = parts[7]
      punctuation = parts[9]
      badword = parts[10]
      notConsistent = parts[12]
      # result = 'Pass!' if length=='False' and badword =='False'and notConsistent =='False' else 'NotPass!'
      print(sku + '\t' +  sku2res[sku])
    else:
      print(sku)


def analyze_training_data_puncuation_distribution(review_log_file):
  cnt = {}
  with open(review_log_file, 'r', encoding='utf-8') as f:
    while (True):
      line = f.readline()
      if (not line):
        break
      line = line.strip()
      parts = line.split('\t')
      label = parts[1]
      cat = parts[4]
      key = label + " " + cat
      if cat not in cnt:
        cnt[cat] = {}
      if label not in cnt[cat]:
        cnt[cat][label] = 0
      cnt[cat][label] += 1
  final_cnt = sorted(cnt.items(), key=lambda x: sum(list(x[1].values())), reverse=True)
  p = 'Pass!'
  np = 'NotPass!'
  for cat, dic in final_cnt:
    print('{0}: Pass:{1}\t NotPass:{2}'.format(cat, dic[p] if p in dic else 0, dic[np] if np in dic else 0))


def word_level_punctuation_rewriting(review_log_file):
  # period_pattern = r'[^ ]+ [。;！] [^ ]+'
  period_pattern = r'[。;！] [^ ]+'
  comma_pattern = r'[，] [^ ]+'
  puncTokensCnt = {}
  commaTokensCnt = {}

  with open(review_log_file, 'r', encoding='utf-8') as f:
    while (True):
      line = f.readline()
      if (not line):
        break
      line = line.strip()
      parts = line.split('\t')
      label = parts[1]
      cat = parts[4]
      writing = parts[5]
      tokenized_writing = parts[6]
      lengthError = parts[7]
      punctuationError = parts[9]
      badWordError = parts[10]
      notConsistentError = parts[12]
      period_context_array = re.findall(period_pattern, tokenized_writing)
      comma_context_array = re.findall(comma_pattern, tokenized_writing)

      if lengthError == 'False' and badWordError == 'False' and notConsistentError == 'False':
        for period_context in period_context_array:
          period_context = period_context[2:]
          if cat not in puncTokensCnt:
            puncTokensCnt[cat] = {}
          if period_context not in puncTokensCnt[cat]:
            puncTokensCnt[cat][period_context] = [0, 0]
          puncTokensCnt[cat][period_context][0] += 1

      if lengthError == 'False' and badWordError == 'False' and notConsistentError == 'False':
        for comma_context in comma_context_array:
          comma_context = comma_context[2:]
          if cat in puncTokensCnt and comma_context in puncTokensCnt[cat]:
            puncTokensCnt[cat][comma_context][1] += 1

  final_cnt = sorted(puncTokensCnt.items(), key=lambda x: sum([pair[0] for pair in list(x[1].values())]), reverse=True)
  final_cnt = [(cat, sorted(v.items(), key=lambda y: y[1][0], reverse=True)) for cat, v in final_cnt]
  for cat, dic in final_cnt:
    print('{0}:'.format(cat))
    for w, f in dic:
      print('{0}:{1},{2}'.format(w, f[0], f[1]))
    print('\n')


def get_ppl(file):
  scores = []
  with open(file, 'r', encoding='utf-8') as f:
    while(True):
      line = f.readline()
      if not line:
        break
      line = line.strip()
      if ('avg_log_p: ' in line):
        s = float(line[line.index('avg_log_p: ') + len('avg_log_p: '):])
        scores.append(s)
      if ('ERROR' in line):
        scores.append(-math.inf)
  assert len(scores) == 313
  with open(file+'.score', 'w', encoding='utf-8') as w:
    for s in scores:
      w.write(str(s) + '\n')


if __name__ == '__main__':
  #root = r'/export/homes/baojunwei/alpha-w-pipeline'
  root = sys.argv[1]
  domain = sys.argv[2] #'chuju'
  dataset = sys.argv[3] #'test'
  ckg_path = os.path.join(root, 'data', 'ckg', domain, 'ckg0', '*')
  blackWord_path = os.path.join(root, 'data', 'blackList.txt')
  ckg = CKG(ckg_path)
  black_list = BlackWords(blackWord_path)

  if (False):  # test
    # sku_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi/test.sku'
    # kb_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi/test.table'
    sku_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\test.sku'
    kb_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\test.table'
    discriminator = ConsistentDiscriminator(ckg, sku_file, kb_file)

    predictin_file = r'/home/baojunwei/project/alpha_writing/log/jiayongdianqi/kb2writing_t2s_bigVocab/decode_test_20maxenc_4beam_20mindec_50maxdec_force_decoding_ckpt-561730/decoded/decoded.txt'
    # log_File = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.review.v0.1.3.log'
    # judge_file(sku_file, predictin_file, discriminator, black_list, log_File)
    print('===============================')

    predictin_file = r'/home/baojunwei/project/alpha_writing/log/jiayongdianqi/kb2writing_t2s_bigVocab/decode_test_20maxenc_4beam_20mindec_50maxdec_force_decoding_ckg_ckpt-561730/decoded/decoded.txt'
    # log_File = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.review.v0.1.3.log'
    # judge_file(sku_file, predictin_file, discriminator, black_list, log_File)
    print('===============================')

    predictin_file = r'/home/baojunwei/project/alpha_writing/log/jiayongdianqi/kb2writing_t2s_bigVocab/decode_test_20maxenc_10beam_20mindec_50maxdec_force_decoding_ckg_ckpt-561730/decoded/decoded.txt'
    # log_File = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.review.v0.1.3.log'
    # judge_file(sku_file, predictin_file, discriminator, black_list, log_File)
    print('===============================')

    # predictin_file = r'/home/baojunwei/project/alpha_writing/log/jiayongdianqi/kb2writing_t2s_bigVocab/decode_test_20maxenc_10beam_20mindec_50maxdec_force_decoding_ckg_gridBS_2sellPoint_ckpt-561730/decoded/decoded.txt'
    predictin_file = r'D:\Research\Projects\AlphaWriting\model\data\test.prediction.t2s_forceDecoding-ckg-gridBS-KB0.1.5'
    # log_File = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.review.v0.1.3.log'
    judge_file(sku_file, predictin_file, discriminator, black_list, log_File)
    print('===============================')

  if (False):  # xunapin
    # sku_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin/xuanpin.merge.sku'
    # kb_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin/xuanpin.merge.sku.table'
    sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.merge.sku'
    kb_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.merge.sku.table'
    discriminator = ConsistentDiscriminator(ckg, sku_file, kb_file)

    # predictin_file = r'/home/baojunwei/project/alpha_writing/log/jiayongdianqi/kb2writing_t2s_bigVocab/decode_xuanpin_20maxenc_4beam_20mindec_50maxdec_force_decoding_ckpt-561730/decoded/decoded.txt'
    predictin_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.merge.sku.prediction.writing.v0.1.3'
    log_File = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.review.v0.1.3.log'
    # judge_file(sku_file, predictin_file, discriminator, black_list, log_File)
    print('===============================')

    # predictin_file = r'/home/baojunwei/project/alpha_writing/log/jiayongdianqi/kb2writing_t2s_bigVocab/decode_xuanpin_20maxenc_4beam_20mindec_50maxdec_force_decoding_ckg_ckpt-561730/decoded/decoded.txt'
    predictin_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.merge.sku.prediction.writing.v0.1.4'
    log_File = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.review.v0.1.4.log'
    # judge_file(sku_file, predictin_file, discriminator, black_list, log_File)
    print('===============================')

    # predictin_file = r'/home/baojunwei/project/alpha_writing/log/jiayongdianqi/kb2writing_t2s_bigVocab/decode_xuanpin_20maxenc_10beam_20mindec_50maxdec_force_decoding_ckg_ckpt-561730/decoded/decoded.txt'
    predictin_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.merge.sku.prediction.writing.v0.1.5.ocr_title_sps.noNull.beam4'
    log_File = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.review.v0.1.5.ocr_title_sps.noNull.beam4.log'
    # judge_file(sku_file, predictin_file, discriminator, black_list, log_File)
    print('===============================')

    # predictin_file = r'/home/baojunwei/project/alpha_writing/log/jiayongdianqi/kb2writing_t2s_bigVocab/decode_xuanpin_20maxenc_4beam_20mindec_50maxdec_force_decoding_ckg_gridBS_2sellPoint_ckpt-561730/decoded/decoded.txt'
    predictin_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.merge.sku.prediction.writing.v0.1.5.ocr_sps.noNull.beam4'
    log_File = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.review.v0.1.5.ocr_sps.noNull.beam4.log'
    # judge_file(sku_file, predictin_file, discriminator, black_list, log_File)
    print('===============================')

    # predictin_file = r'/home/baojunwei/project/alpha_writing/log/jiayongdianqi/kb2writing_t2s_bigVocab/decode_xuanpin_20maxenc_4beam_20mindec_50maxdec_force_decoding_ckg_gridBS_2sellPoint_ckpt-561730/decoded/decoded.txt'
    predictin_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.merge.sku.prediction.writing.v0.1.5.ocr_sps.noNull.beam4_blackList2'
    log_File = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.review.log'
    judge_file(sku_file, predictin_file, discriminator, black_list, log_File)
    print('===============================')

  if (False):  # xunapin2
    # sku_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin/xuanpin.merge.sku'
    # kb_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin/xuanpin.merge.sku.table'
    sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.sku'
    kb_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.table'
    discriminator = ConsistentDiscriminator(ckg, sku_file, kb_file)

    # predictin_file = r'/home/baojunwei/project/alpha_writing/log/jiayongdianqi/kb2writing_t2s_bigVocab/decode_xuanpin_20maxenc_4beam_20mindec_50maxdec_force_decoding_ckg_gridBS_2sellPoint_ckpt-561730/decoded/decoded.txt'
    predictin_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.prediction.writing.KB0.1.5'
    log_File = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.review.log'
    judge_file(sku_file, predictin_file, discriminator, black_list, log_File)
    print('===============================')

  if (False):  # xunapin3
    # sku_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin/xuanpin.merge.sku'
    # kb_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin/xuanpin.merge.sku.table'
    sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.validSku'
    kb_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.table'
    discriminator = ConsistentDiscriminator(ckg, sku_file, kb_file)

    # predictin_file = r'/home/baojunwei/project/alpha_writing/log/jiayongdianqi/kb2writing_t2s_bigVocab/decode_xuanpin_20maxenc_4beam_20mindec_50maxdec_force_decoding_ckg_gridBS_2sellPoint_ckpt-561730/decoded/decoded.txt'
    predictin_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.prediction.writing.KB0.1.5'
    log_File = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.review'
    bad_sku_File = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.sku.bad'
    sku2res = judge_file(sku_file, predictin_file, discriminator, black_list, log_File, bad_sku_File)
    print('===============================')

    target_skus = [_.strip() for _ in open(r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.skus.300', 'r', encoding='utf-8')]
    print_discriminater_res_for_specific_skus(target_skus, sku2res)

  if (False):  # xunapin4
    # sku_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin/xuanpin.merge.sku'
    # kb_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin/xuanpin.merge.sku.table'
    sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\xuanpin4.validSku'
    kb_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\xuanpin4.table'
    discriminator = ConsistentDiscriminator(ckg, sku_file, kb_file)

    spNum = 3
    root = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\sp' + str(spNum)
    # predictin_file = os.path.join(root, r'xuanpin4.prediction.writing.punc.KB0.1.5')
    # log_File = os.path.join(root, r'xuanpin4.punc.review')
    # bad_sku_File = os.path.join(root, r'xuanpin4.sku.bad')
    predictin_file = os.path.join(root, r'xuanpin4.prediction.writing.KB0.1.5')
    log_File = os.path.join(root, r'xuanpin4.review')
    bad_sku_File = os.path.join(root, r'xuanpin4.sku.bad')
    sku2res = judge_file(sku_file, predictin_file, discriminator, black_list, log_File, bad_sku_File)
    print('===============================')

    target_skus = [_.strip() for _ in open(r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\xuanpin4.validSku', 'r', encoding='utf-8')]
    print_discriminater_res_for_specific_skus(target_skus, sku2res)

  if (False):  # xunapin5
    # sku_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin/xuanpin.merge.sku'
    # kb_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin/xuanpin.merge.sku.table'
    sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.validSku'
    kb_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.table'
    discriminator = ConsistentDiscriminator(ckg, sku_file, kb_file)

    spNum = 3
    root = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\sp' + str(spNum)
    predictin_file = os.path.join(root, r'xuanpin3.prediction.writing.punc.KB0.1.5')
    # predictin_file = os.path.join(root, r'xuanpin3.prediction.writing.KB0.1.5')

    log_File = os.path.join(root, r'xuanpin3.punc.review')
    # log_File = os.path.join(root, r'xuanpin3.review')
    bad_sku_File = os.path.join(root, r'xuanpin3.sku.bad')
    sku2res = judge_file(sku_file, predictin_file, discriminator, black_list, log_File, bad_sku_File)
    print('===============================')

    target_skus = [_.strip() for _ in
                   open(r'D:\Research\Projects\AlphaWriting\model\data\xuanpin5\xuanpin5.sku',
                        'r', encoding='utf-8')]
    print_discriminater_res_for_specific_skus(target_skus, sku2res)

  if (False):  # xuanpin6
    # sku_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.validSku')
    # kb_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.validSku')
    sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin6\xuanpin6.validSku'
    kb_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin6\xuanpin6.table'
    discriminator = ConsistentDiscriminator(ckg, sku_file, kb_file)

    spNum = 3
    root = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin6\sp' + str(spNum)
    # predictin_file = os.path.join(root, r'xunapin6.prediction.writing.punc.KB0.1.5')
    # log_File = os.path.join(root, r'xunapin6.punc.review')
    # bad_sku_File = os.path.join(root, r'xunapin6.sku.bad')
    predictin_file = os.path.join(root, r'xuanpin6.prediction.writing.punc.KB0.1.5')
    log_File = os.path.join(root, r'xuanpin6.review')
    bad_sku_File = os.path.join(root, r'xuanpin6.sku.bad')
    sku2res = judge_file(sku_file, predictin_file, discriminator, black_list, log_File, bad_sku_File)
    print('===============================')

    target_skus = [_.strip() for _ in open(r'D:\Research\Projects\AlphaWriting\model\data\xuanpin6\xuanpin6.validSku', 'r', encoding='utf-8')]
    print_discriminater_res_for_specific_skus(target_skus, sku2res)

  if (True):  # chuju
    sku_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.validSku')
    kb_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.table')
    # sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin6\xuanpin6.validSku'
    # kb_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin6\xuanpin6.table'
    discriminator = ConsistentDiscriminator(ckg, sku_file, kb_file)


    predictin_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + r'.prediction.writing')
    log_File = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.review')
    bad_sku_File = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.sku.bad')
    sku2res = judge_file(sku_file, predictin_file, discriminator, black_list, log_File, bad_sku_File)
    # print('===============================')

    # target_skus = [_.strip() for _ in open(r'D:\Research\Projects\AlphaWriting\model\data\xuanpin6\xuanpin6.validSku', 'r', encoding='utf-8')]
    # print_discriminater_res_for_specific_skus(target_skus, sku2res)

  if (False):  # alphaSales
    # sku_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin/xuanpin.merge.sku'
    # kb_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin/xuanpin.merge.sku.table'
    sku_file = r'D:\Research\Projects\AlphaWriting\model\data\alphaSales\alphaSales.validSku'
    kb_file = r'D:\Research\Projects\AlphaWriting\model\data\alphaSales\alphaSales.table'
    discriminator = ConsistentDiscriminator(ckg, sku_file, kb_file)

    spNum = 3
    root = r'D:\Research\Projects\AlphaWriting\model\data\alphaSales\sp' + str(spNum)
    predictin_file = os.path.join(root, r'alphaSales.prediction.writing.punc.KB0.1.5')
    log_File = os.path.join(root, r'alphaSales.punc.review')
    bad_sku_File = os.path.join(root, r'alphaSales.sku.bad')
    # predictin_file = os.path.join(root, r'alphaSales.prediction.writing.KB0.1.5')
    # log_File = os.path.join(root, r'alphaSales.review')
    # bad_sku_File = os.path.join(root, r'alphaSales.sku.bad')
    sku2res = judge_file(sku_file, predictin_file, discriminator, black_list, log_File, bad_sku_File)
    print('===============================')

    target_skus = [_.strip() for _ in open(r'D:\Research\Projects\AlphaWriting\model\data\alphaSales\alphaSales.validSku', 'r', encoding='utf-8')]
    print_discriminater_res_for_specific_skus(target_skus, sku2res)

  if (False):  # review
    sku_file = r'D:\Research\Projects\AlphaWriting\model\data\writingReview\review.sku.validSkus'
    kb_file = r'D:\Research\Projects\AlphaWriting\model\data\writingReview\review.sku.table'
    discriminator = ConsistentDiscriminator(ckg, sku_file, kb_file)
    predictin_file = r'D:\Research\Projects\AlphaWriting\model\data\writingReview\review.sku.validWriting'
    log_File = r'D:\Research\Projects\AlphaWriting\model\data\writingReview\review.review.log'

    judge_file(sku_file, predictin_file, discriminator, black_list, log_File)
    print('===============================')

  if (False):  # review-trainingData
    sku_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\train.sku'
    kb_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\train.table'
    discriminator = ConsistentDiscriminator(ckg, sku_file, kb_file)

    root = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi'
    predictin_file = os.path.join(root, r'train.text')
    log_File = os.path.join(root, r'train.review.log')
    bad_sku_File = os.path.join(root, r'train.sku.bad')
    sku2res = judge_file(sku_file, predictin_file, discriminator, black_list, log_File, bad_sku_File)
    print('===============================')

  if (False):  # review-devData
    sku_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\dev.sku'
    kb_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\dev.table'
    discriminator = ConsistentDiscriminator(ckg, sku_file, kb_file)

    root = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi'
    predictin_file = os.path.join(root, r'dev.text')
    log_File = os.path.join(root, r'dev.review.log')
    bad_sku_File = os.path.join(root, r'dev.sku.bad')
    sku2res = judge_file(sku_file, predictin_file, discriminator, black_list, log_File, bad_sku_File)
    print('===============================')

  if (False):  # review-testData
    sku_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\test.sku'
    kb_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\test.table'
    discriminator = ConsistentDiscriminator(ckg, sku_file, kb_file)

    root = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi'
    predictin_file = os.path.join(root, r'test.text')
    log_File = os.path.join(root, r'test.review.log')
    bad_sku_File = os.path.join(root, r'test.sku.bad')
    sku2res = judge_file(sku_file, predictin_file, discriminator, black_list, log_File, bad_sku_File)
    print('===============================')

  if (False):
    log_File = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\train.review.log.pure'
    # analyze(log_File)
    word_level_punctuation_rewriting(log_File)

  if (False):
    file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\sp5\2019-06-12_23_21_09.log'
    get_ppl(file)

    file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\sp4\2019-06-12_23_22_49.log'
    get_ppl(file)

    file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\sp3\2019-06-12_23_24_11.log'
    get_ppl(file)

    file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\sp2\2019-06-12_23_24_52.log'
    get_ppl(file)

    file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\sp1\2019-06-12_23_25_39.log'
    get_ppl(file)
