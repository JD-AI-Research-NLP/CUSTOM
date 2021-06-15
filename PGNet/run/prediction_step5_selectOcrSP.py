import os
from collections import Counter
import re
import sys

class BlackWords(object):
  def __init__(self, blackWord_file):
    with open(blackWord_file, 'r', encoding='utf-8') as f:
      self.black_list = [_.strip() for _ in f.readlines()]
  def getWords(self):
    return self.black_list

def sellingPointSelection(file, black_list, spNum, largeIdf):
  sellingPoints = []
  with open(file, 'r', encoding='utf-8') as fin:
    # with open(file + '.constraint', 'w', encoding='utf-8') as fw:
    sps = []
    while (True):
      line = fin.readline()
      if not line:
        break
      line = line.strip()
      line = line.replace('【', '').replace('】', '').replace('[', '').replace(']', '') \
        .replace('(', '').replace(')', '').replace('⑥', '').replace('①', '')
      parts = line.split('\t')
      if line == '':
        sps = sorted(sps, key=lambda x: x[1], reverse=largeIdf)
        sps = sps[:spNum] if len(sps) > spNum else sps
        sellingPoints.append('\t'.join([_[0] for _ in sps]))
        # fw.write('\t'.join([_[0] for _ in sps]) + '\n')
        sps = []
      elif len(parts) >= 2:
        k = parts[0].strip()
        for i in range(1, len(parts)):
          vs = parts[i].strip().split(':')
          v = [_ for _ in re.split(r'[.+]', vs[0]) if isGood(_, black_list)]
          if len(v) > 0:
            s = vs[1]
            sps.append((v[0], s))
            break
  return sellingPoints


def sellingPointSelectionNew(file, black_list, spNum, largeIdf):
  sku2sellingPoints = {}
  with open(file, 'r', encoding='utf-8') as fin:
    # with open(file + '.constraint', 'w', encoding='utf-8') as fw:
    sps = []
    sku = ''
    while (True):
      line = fin.readline()
      if not line:
        break
      line = line.strip()
      line = line.replace('【', '').replace('】', '').replace('[', '').replace(']', '') \
        .replace('(', '').replace(')', '')\
        .replace('①', '').replace('②', '').replace('③', '').replace('④', '').replace('⑤', '')\
        .replace('⑥', '').replace('⑦', '').replace('⑧', '').replace('⑨', '').replace('⑩', '')
      parts = line.split('\t')
      if line == '':
        sps = sorted(sps, key=lambda x: x[1], reverse=largeIdf)
        sps = sps[:spNum] if len(sps) > spNum else sps
        sku2sellingPoints[sku]='\t'.join([_[0] for _ in sps])
        # fw.write('\t'.join([_[0] for _ in sps]) + '\n')
        sps = []
        sku = ''
      elif line.startswith('https'):
        sku = line.replace(r'https://item.jd.com/', '').replace('.html', '') #https://item.jd.com/3989011.html
      elif len(parts) >= 2:
        k = parts[0].strip()
        for i in range(1, len(parts)):
          vs = parts[i].strip().split(':')
          v = [_ for _ in re.split(r'[.+]', vs[0]) if isGood(_, black_list)]
          if len(v) > 0:
            s = vs[1]
            sp = v[0]
            sp = sp.strip('-o*!')
            charCount = Counter(sp)
            if charCount['”'] != charCount['“']:
              sp = sp.replace('”', '').replace('“', '')
            sp=sp.replace('赫鄂', '赫兹')

            sps.append((sp, s))
            break
  return sku2sellingPoints

def isGood(str, black_list):
  nonChineseStrPattern = r'^[^\u4e00-\u9fa5]+$'
  endWithDigitPattern = r'[0-9]+$'
  if len(str) <= 2 or len(str) > 10:
    return False
  if '什么' in str:
    return False
  if '家用' in str:
    return False
  if "商用" in str:
    return False
  if '送' in str:
    return False
  if '选择' in str:
    return False
  if '京东' in str:
    return False
  if '更多' in str:
    return False
  if '经销商' in str:
    return False
  if any([w in str for w in black_list.getWords()]):
    return False
  if re.match(nonChineseStrPattern, str):
    return False
  if re.search(endWithDigitPattern, str):
    return False
  return True


def write2file(skus, file, sku2sellingPoints):
  with open(file, 'w', encoding='utf-8') as f:
    for sku in skus:
      if sku in sku2sellingPoints:
        f.write(sku2sellingPoints[sku] + '\n')
      else:
        f.write('\n')


def loadSkus(file):
  with open(file, 'r', encoding='utf-8') as f:
    skus = [sku.strip().split('\t')[0] for sku in f.readlines()]
  return skus

def loadSingleLineFile(file):
  with open(file, 'r', encoding='utf-8') as f:
    skus = [sku.strip() for sku in f.readlines()]
  return skus


def sellingPointsForTestData(spInFile, black_list, skuFile5cat, skuFileAllCat):
  sps5cat = sellingPointSelection(spInFile, black_list)
  skus5cat = loadSkus(skuFile5cat)
  sku2sps = {}
  for id in range(len(skus5cat)):
    sp = sps5cat[id]
    sku = skus5cat[id]
    sku2sps[sku] = sp

  skusAllCat = loadSkus(skuFileAllCat)
  spsAllCat = []
  for sku in skusAllCat:
    sps = ''
    if sku in sku2sps:
      sps = sku2sps[sku]
    spsAllCat.append(sps)
  return spsAllCat


def dump5CatPrediction(predFileAllCat, darenWritingFileAllCat, constraintFileAllCat, skuFileAllCat,
                       skuFile5Cat, predFile5Cat, darenWritingFile5Cat, constraintFile5Cat):
  skuAllCat = loadSkus(skuFileAllCat)
  predAllCat = loadSingleLineFile(predFileAllCat)
  darenAllCat = loadSingleLineFile(darenWritingFileAllCat)
  constAllCat = loadSingleLineFile(constraintFileAllCat)

  sku5Cat = loadSkus(skuFile5cat)
  sku2PredAndDaren = {}
  for i in range(len(skuAllCat)):
    sku = skuAllCat[i]
    pred = predAllCat[i]
    daren = darenAllCat[i]
    constraint = constAllCat[i]
    sku2PredAndDaren[sku] = (pred, daren, constraint)

  with open(predFile5Cat, 'w', encoding='utf-8') as fp:
    with open(darenWritingFile5Cat, 'w', encoding='utf-8') as fd:
      with open(constraintFile5Cat, 'w', encoding='utf-8') as fc:
        for sku in sku5Cat:
          tup = sku2PredAndDaren[sku]
          pred = tup[0]
          daren = tup[1]
          const = tup[2]
          fp.write(pred + '\n')
          fd.write(daren + '\n')
          fc.write(const + '\n')


if __name__ == '__main__':

  #root = r'/export/homes/baojunwei/alpha-w-pipeline/data'
  root = os.path.join(sys.argv[1], 'data')
  domain = sys.argv[2]#'chuju'
  dataset = sys.argv[3]#'test'
  black_word_list = os.path.join(root, 'blackList.txt')
  black_list = BlackWords(black_word_list)

  if (False): #xuanpin-1
    inFile = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.merge.sku.sellingPoints.fromYifan'
    outFile = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.merge.sku.constraint.new-2sellingPoints'
    sps = sellingPointSelection(inFile, black_list)
    write2file(outFile, sps)

  if (False): #xuanpin-2
    inFile = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.title.sellingPoint.fromYiFan'
    outFile = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.constraint-2sellingPoints'
    sps = sellingPointSelection(inFile, black_list)
    write2file(outFile, sps)

  if (False): #xuanpin-3
    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.skus.300'
    largeIdf = True
    idfLabel = 'largeIdf' if largeIdf else 'smallIdf'
    src = 'ocr'#'title'
    for spNum in range(0, 6):
      inFile = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.sellingPoint.ocr.fromYinfan'
      outFile = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.constraint-'+str(spNum)+\
                'sellingPoints-{0}-{1}'.format(src, idfLabel)
      sku2sps = sellingPointSelectionNew(inFile, black_list, spNum, largeIdf)
      skus = list(sku2sps.keys())#all sku
      # skus = [sku.strip() for sku in open(valid_sku_file, 'r', encoding='utf-8')]
      write2file(skus, outFile, sku2sps)

  if (False): #xuanpin-4
    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\xuanpin4.validSku'
    largeIdf = True
    idfLabel = 'largeIdf' if largeIdf else 'smallIdf'
    src = 'ocr'#'title'
    for spNum in range(0, 6):
      inFile = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\xuanpin4.sellingPoint.ocr.fromYinfan'
      outFile = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\xuanpin4.constraint-'+str(spNum)+\
                'sellingPoints-{0}-{1}'.format(src, idfLabel)
      sku2sps = sellingPointSelectionNew(inFile, black_list, spNum, largeIdf)
      # skus = list(sku2sps.keys())#all sku
      skus = [sku.strip() for sku in open(valid_sku_file, 'r', encoding='utf-8')]
      write2file(skus, outFile, sku2sps)

  if (False): #xuanpin-6
    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin6\xuanpin6.validSku'
    largeIdf = True
    idfLabel = 'largeIdf' if largeIdf else 'smallIdf'
    src = 'ocr'#'title'
    for spNum in range(0, 6):
      inFile = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin6\xuanpin6.sellingPoint.ocr.fromYinfan'
      outFile = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin6\xuanpin6.constraint-'+str(spNum)+\
                'sellingPoints-{0}-{1}'.format(src, idfLabel)
      sku2sps = sellingPointSelectionNew(inFile, black_list, spNum, largeIdf)
      # skus = list(sku2sps.keys())#all sku
      skus = [sku.strip() for sku in open(valid_sku_file, 'r', encoding='utf-8')]
      write2file(skus, outFile, sku2sps)

  if (False): #AlphaSales
    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\alphaSales\alphaSales.validSku'
    largeIdf = True
    idfLabel = 'largeIdf' if largeIdf else 'smallIdf'
    src = 'ocr'#'title'
    for spNum in range(0, 6):
      inFile = r'D:\Research\Projects\AlphaWriting\model\data\alphaSales\alphaSales.sellingPoint.ocr.fromYinfan'
      outFile = r'D:\Research\Projects\AlphaWriting\model\data\alphaSales\alphaSales.constraint-'+str(spNum)+\
                'sellingPoints-{0}-{1}'.format(src, idfLabel)
      sku2sps = sellingPointSelectionNew(inFile, black_list, spNum, largeIdf)
      # skus = list(sku2sps.keys())#all sku
      skus = [sku.strip() for sku in open(valid_sku_file, 'r', encoding='utf-8')]
      write2file(skus, outFile, sku2sps)

  if (True): #chuju

    valid_sku_file = os.path.join(root, domain + '_' + dataset, dataset + '.validSku')
    largeIdf = True
    idfLabel = 'largeIdf' if largeIdf else 'smallIdf'
    src = 'ocr'#'title'
    for spNum in range(0, 6):
      inFile = os.path.join(root, domain + '_' + dataset, dataset + '.sp')
      outFile = os.path.join(root, domain + '_' + dataset, dataset + '.constraint-'+str(spNum)+\
                'sellingPoints-{0}-{1}'.format(src, idfLabel))
      sku2sps = sellingPointSelectionNew(inFile, black_list, spNum, largeIdf)
      # skus = list(sku2sps.keys())#all sku
      skus = [sku.strip() for sku in open(valid_sku_file, 'r', encoding='utf-8')]
      write2file(skus, outFile, sku2sps)

  if (False): #test/dev set
    dataset = 'dev'
    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\filtered_data\\'+dataset+r'.sku'
    largeIdf = True
    idfLabel = 'largeIdf' if largeIdf else 'smallIdf'
    # src = 'ocr'#'title'
    src = 'title'#'ocr'

    for spNum in range(0, 6):
      inFile = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\filtered_data\\'+dataset+'_'+src+'_salepoint'
      outFile = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\filtered_data\\'+dataset+'.constraint-'+str(spNum)+\
                'sellingPoints-{0}-{1}'.format(src, idfLabel)
      sku2sps = sellingPointSelectionNew(inFile, black_list, spNum, largeIdf)
      # skus = list(sku2sps.keys())#all sku
      skus = [sku.strip() for sku in open(valid_sku_file, 'r', encoding='utf-8')]
      write2file(skus, outFile, sku2sps)

  if (False): #post processing
    predictionFileAllCat = r'D:\Research\Projects\AlphaWriting\model\data\test.prediction.t2s_forceDecoding-ckg-gridBS-KB0.1.5'
    darenWritingFileAllCat = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\test.text'
    skuFileAllCat = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\test.sku'
    constraintFileAllCat = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\test.constraint.new-2sellingPoints'
    skuFile5cat = r'D:\Research\Projects\AlphaWriting\model\data\5category\test.sku.top5Cat'
    predictionFile5Cat = r'D:\Research\Projects\AlphaWriting\model\data\test.prediction.t2s_forceDecoding-ckg-gridBS-5cat-KB0.1.5'
    darenWritingFile5Cat = r'D:\Research\Projects\AlphaWriting\model\data\5category\test.text.top5Cat'
    constraintFile5Cat = r'D:\Research\Projects\AlphaWriting\model\data\5category\test.constraint.new-2sellingPoints.top5Cat'

    dump5CatPrediction(predictionFileAllCat, darenWritingFileAllCat, constraintFileAllCat, skuFileAllCat,
                       skuFile5cat, predictionFile5Cat, darenWritingFile5Cat, constraintFile5Cat)

  #==============================================
  if(False):
    dataset = 'train'
    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\filter_data\\'+dataset+'.sku'
    valid_table_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\filter_data\\'+dataset+'.table'

    for spNum in range(0, 6):
      inFile = r'D:\Research\Projects\AlphaWriting\model\data\alphaSales\alphaSales.sellingPoint.ocr.fromYinfan'
      outFile = valid_table_file + '.withSP'

      sku2sps = sellingPointSelectionNew(inFile, black_list, spNum, largeIdf)
      # skus = list(sku2sps.keys())#all sku
      skus = [sku.strip() for sku in open(valid_sku_file, 'r', encoding='utf-8')]
      write2file(skus, outFile, sku2sps)
