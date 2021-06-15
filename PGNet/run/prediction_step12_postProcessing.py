import re
import os
import sys
pattern = r'^[^\u4e00-\u9fa5]+$'


def loadAllData(file):
  pairs = {}
  with open(file, 'r', encoding='utf-8') as f:
    id = 0
    text = ""
    kb = {}
    while 1:
      line = f.readline()
      if not line:
        print(id)
        break
      # if id > 100000:
      #   break

      line = line.strip('\n')
      if line.startswith('AlphaWriting'):
        id = int(line.split('-')[1])
      if line.startswith('Title:'):
        title = line[7:]
      if line.startswith('Text:'):
        text = line[6:]
      if line.startswith('Sku:'):
        sku = line[5:]
      if line.startswith('KB:'):
        kv = line[4:].split(' ||| ')
        key = kv[0].strip()
        value = kv[1].strip()
        if (len(kv) == 2 and key != "" and value != "" and value != "--" and key != "Lables" and key != "Labels"):
          if not kb.get(key, None):
            kb[key] = []
          kb[key].append(value)
      if line == "":
        if text != "" and len(kb.keys()) != 0:
          if not pairs.get(sku, None):
            pairs[sku] = []
          pairs[sku].append((id, kb, text, title))
        else:
          print(id)
        kb = {}
        text = ""

  return pairs


def loadSkus(test_sku_file):
  with open(test_sku_file, 'r', encoding='utf-8') as skusSr:
    skus = [_.strip() for _ in skusSr.readlines()]
    print(len(skus))

  return skus


def add_kb(sku_list, file, id2kv):
  # file = r"/home/baojunwei/notespace/tensorflow-models/data_JD/kb/jd.kb"
  with open(file, mode='r', encoding='utf-8') as f_in:
    cnt = 0
    errorCnt = 0
    for line in f_in:
      parts = line.strip('\n ').split('\t')
      sbj = parts[0]
      pred = parts[1]
      obj = parts[2]

      if (sbj in sku_list):
        if sbj not in id2kv:
          errorCnt += 1
          continue
        if pred not in id2kv[sbj]:
          id2kv[sbj][pred] = []
        id2kv[sbj][pred].append(obj)
        cnt += 1
        if (cnt % 10 == 0):
          print("Loading %d lines.\r" % cnt)
  print("KB Length: %d\n" % len(id2kv))
  print("ErrorCnt: %d\n" % errorCnt)

  return id2kv

def load_kb_for_dataset(sku_info_file, table_file):
  sku2info = {}
  schema = []
  skus = []
  with open(sku_info_file, 'r', encoding='utf-8') as f:
    schema = f.readline().strip().split('\t')
    while(True):
      line = f.readline()
      if not line:
        break
      line = line.strip()
      parts = line.split('\t')
      sku = parts[0]
      skus.append(sku)
      sku2info[sku] = {}
      for i in range(len(schema)):
        pred = schema[i]
        sku2info[sku][pred] = [parts[i]]

  index = 0
  with open(table_file, 'r', encoding='utf-8') as f:
    while(True):
      line = f.readline()
      if not line:
        break
      line = line.strip()
      sku = skus[index]
      kvs = [_.split(' $$ ') for _ in line.split(' ## ')]
      for k, v in kvs:
        if k in schema:
          continue
        if k not in sku2info[sku]:
          sku2info[sku][k] = []
        sku2info[sku][k].append(v)
      index += 1
  print("KB Length: %d\n" % len(sku2info))
  return sku2info

def add_sku_basic_info_back(sku_basic_info_file, kv):
  with open(sku_basic_info_file, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      parts = line.split('\t')
      Sku = parts[0]
      SkuName = parts[1]
      Brand_id = parts[2]
      BrandName_en = parts[3]
      BrandName_cn = parts[4]
      BrandName_full = parts[5]
      CategoryL1_id = parts[6]
      CategoryL1 = parts[7]
      CategoryL2_id = parts[8]
      CategoryL2 = parts[9]
      CategoryL3_id = parts[10]
      CategoryL3 = parts[11]
      kv[Sku] = []
      kv[Sku].append(('Sku', Sku))
      kv[Sku].append(('SkuName', SkuName))
      kv[Sku].append(('Brand_id', Brand_id))
      kv[Sku].append(('BrandName_en', BrandName_en))
      kv[Sku].append(('BrandName_cn', BrandName_cn))
      kv[Sku].append(('BrandName_full', BrandName_full))
      kv[Sku].append(('CategoryL1_id', CategoryL1_id))
      kv[Sku].append(('CategoryL1', CategoryL1))
      kv[Sku].append(('CategoryL2_id', CategoryL2_id))
      kv[Sku].append(('CategoryL2', CategoryL2))
      kv[Sku].append(('CategoryL3_id', CategoryL3_id))
      kv[Sku].append(('CategoryL3', CategoryL3))
    return kv


def add_sku_basic_info(sku_list, sku_basic_info_file, sku2basicInfo):
  with open(sku_basic_info_file, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      parts = line.split('\t')
      Sku = parts[0]
      SkuName = parts[1]
      Brand_id = parts[2]
      BrandName_en = parts[3]
      BrandName_cn = parts[4]
      BrandName_full = parts[5]
      CategoryL1_id = parts[6]
      CategoryL1 = parts[7]
      CategoryL2_id = parts[8]
      CategoryL2 = parts[9]
      CategoryL3_id = parts[10]
      CategoryL3 = parts[11]
      if (Sku in sku_list):
        sku2basicInfo[Sku] = {'Sku': [Sku],
                              'SkuName': [SkuName],
                              'Brand_id': [Brand_id],
                              'BrandName_en': [BrandName_en],
                              'BrandName_cn': [BrandName_cn],
                              'BrandName_full': [BrandName_full],
                              'CategoryL1_id': [CategoryL1_id],
                              'CategoryL1': [CategoryL1],
                              'CategoryL2_id': [CategoryL2_id],
                              'CategoryL2': [CategoryL2],
                              'CategoryL3_id': [CategoryL3_id],
                              'CategoryL3': [CategoryL3]}
    return sku2basicInfo


def rule_based_title_generation_forNewSku_andFilter(skus, new_prediction_file, sku2info, sku2prediction, black_list,
                                                    bad_sku_list):
  bad_tokens = ['()', '+', '-']
  with open(new_prediction_file + ".forDebug", 'w', encoding='utf-8') as newPredDebugSw:
    with open(new_prediction_file, 'w', encoding='utf-8') as newPredSw:

      num = 1
      lengthError = 0
      for sku in skus:
        old_title_prediction = sku2prediction[sku][0]
        tokenized_writing = sku2prediction[sku][1].split(' ')
        writing_prediction = sku2prediction[sku][1].replace(' ', '')
        short_sentences = re.split('[，。]', writing_prediction)
        short_sentences = [_ for _ in short_sentences if _ != '']
        short_sentences_len = len(short_sentences)
        writing_len = len(writing_prediction)

        bad_flag = False
        for bad_word in bad_tokens:
          if (bad_word in writing_prediction):
            bad_flag = True
            break
        if (bad_flag):
          continue
        print("pass1")
        if (short_sentences_len <= 3 or short_sentences_len > 10):
          continue
        print("pass2")

        if (sku in bad_sku_list):
          continue
        if (len(tokenized_writing) >= 51):  # max_dec_steps is setup as 50 + 1 (。)
          continue
        bad_flag = False
        for bad_word in black_list:
          if (bad_word in writing_prediction):
            bad_flag = True
            break
        if (bad_flag):
          continue
        if writing_len < 50 or writing_len > 90:
          continue

        old_title_prediction_tokens = old_title_prediction.split(' ')

        if sku not in sku2info:
          if len(old_title_prediction_tokens) <= 3:
            newPredSw.write(old_title_prediction)
          else:
            newPredSw.write(
              old_title_prediction_tokens[0] + " " + ''.join(old_title_prediction_tokens[1:-1]) + " " +
              old_title_prediction_tokens[-1] + '\n')
          continue

        kb = sku2info[sku]
        cat3 = kb['CategoryL3'][0].replace(' ', '').strip()
        brand = kb['BrandName_cn'][0].replace(' ', '').strip()
        brandAll = kb['BrandName_full'][0].replace(' ', '').strip()

        if cat3 == '':
          print(sku)
        if brand == '':
          print('noBrand:', sku)
        if (cat3 in ['冰箱', '净水器', '空调', '洗衣机', '油烟机']):
          specialRecomandation = []

          if (cat3 == '油烟机'):
            specialRecomandation = kb.get('特色推荐', [])

          if (cat3 == '洗衣机'):
            specialRecomandation = kb.get('特色推荐', [])
            if (specialRecomandation == ''):
              tmp = [k.split('_')[1] for k, v in kb.items() if k.startswith('特色功能') and v == '支持']
              if (len([tmp]) > 0):
                specialRecomandation = tmp
            if (('变频' in writing_prediction or '变频' in old_title_prediction) and '主体_电机类型' in kb and '定频' in
                kb['主体_电机类型'][0]):
              writing_prediction = writing_prediction.replace('变频', '定频')
              old_title_prediction = old_title_prediction.replace('变频', '定频')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('变=》定')
            elif (('定频' in writing_prediction or '定频' in old_title_prediction) and '主体_电机类型' in kb and '变频' in
                  kb['主体_电机类型'][0]):
              writing_prediction = writing_prediction.replace('定频', '变频')
              old_title_prediction = old_title_prediction.replace('定频', '变频')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('定=》变')

          if (cat3 == '冰箱'):
            specialRecomandation = kb.get('特色推荐', [])
            if (('变频' in writing_prediction or '变频' in old_title_prediction) and '压缩机' in kb and '定频' in kb['压缩机'][0]):
              writing_prediction = writing_prediction.replace('变频', '定频')
              old_title_prediction = old_title_prediction.replace('变频', '定频')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('变=》定')
            elif (
                ('定频' in writing_prediction or '定频' in old_title_prediction) and '压缩机' in kb and '变频' in kb['压缩机'][0]):
              writing_prediction = writing_prediction.replace('定频', '变频')
              old_title_prediction = old_title_prediction.replace('定频', '变频')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('定=》变')
            if (('风冷' in writing_prediction or '风冷' in old_title_prediction) and '制冷方式' in kb and (
                '风直冷' in kb['制冷方式'][0] or '混冷' in kb['制冷方式'][0])):
              writing_prediction = writing_prediction.replace('风冷', '混冷')
              old_title_prediction = old_title_prediction.replace('风冷', '混冷')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('风=》混')
            elif (('风冷' in writing_prediction or '风冷' in old_title_prediction) and '功能_制冷方式' in kb and (
                '风直冷' in kb['功能_制冷方式'][0] or '混冷' in kb['制冷方式'][0])):
              writing_prediction = writing_prediction.replace('风冷', '混冷')
              old_title_prediction = old_title_prediction.replace('风冷', '混冷')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('风=》混')
          if (cat3 == '净水器'):
            specialRecomandation = kb.get('特色推荐', [])
            if (specialRecomandation == ''):
              specialRecomandation = kb.get('产品特点', [])

          if (cat3 == '空调'):
            specialRecomandation = kb.get('产品特点', [])
            if (('变频' in writing_prediction or '变频' in old_title_prediction) and '变频/定频' in kb and '定频' in kb['变频/定频'][
              0]):
              writing_prediction = writing_prediction.replace('变频', '定频')
              old_title_prediction = old_title_prediction.replace('变频', '定频')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('变=》定')
            elif (('变频' in writing_prediction or '变频' in old_title_prediction) and '功能_变频/定频' in kb and '定频' in
                  kb['功能_变频/定频'][0]):
              writing_prediction = writing_prediction.replace('变频', '定频')
              old_title_prediction = old_title_prediction.replace('变频', '定频')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('变=》定')
            elif (
                ('定频' in writing_prediction or '定频' in old_title_prediction) and '变频/定频' in kb and '变频' in kb['变频/定频'][
              0]):
              writing_prediction = writing_prediction.replace('定频', '变频')
              old_title_prediction = old_title_prediction.replace('定频', '变频')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('定=》变')
            elif (('定频' in writing_prediction or '定频' in old_title_prediction) and '功能_变频/定频' in kb and '变频' in
                  kb['功能_变频/定频'][0]):
              writing_prediction = writing_prediction.replace('定频', '变频')
              old_title_prediction = old_title_prediction.replace('定频', '变频')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('定=》变')

          propertyToks = []
          idx = 0
          while True:
            if (idx >= len(old_title_prediction_tokens)):
              break
            if (old_title_prediction_tokens[idx] not in brandAll):
              break
            idx += 1

          while True:
            if (idx >= len(old_title_prediction_tokens)):
              break
            tmpProperty = ''.join(propertyToks)
            if tmpProperty.startswith(brand.lower()):
              tmpProperty = tmpProperty[len(brand):]
            if (len(set(old_title_prediction_tokens[idx]) & set(cat3)) > 0 and len(tmpProperty) > 1):
              break
            propertyToks.append(old_title_prediction_tokens[idx])
            idx += 1

          src = "None"

          property = ''
          if (True):
            if property == '' and len(propertyToks) > 0:
              property = ''.join(propertyToks)
              src = "FromS2S"
            if property == '' and len(specialRecomandation) > 0:
              property = specialRecomandation[0]
              src = "FromKB"
          if (False):
            if property == '' and len(specialRecomandation) > 0:
              property = specialRecomandation[0]
              src = "FromKB"
            if property == '' and len(propertyToks) > 0:
              property = ''.join(propertyToks)
              src = "FromS2S"

          newCat = ''.join(old_title_prediction_tokens[idx:])
          if property.startswith(brand.lower()):
            property = property[len(brand):]
          if property.endswith(cat3.lower()):
            property = property[:len(property) - len(cat3)]
          if property.endswith(newCat.lower()):
            property = property[:len(property) - len(newCat)]

          # if (property == ""):
          #   print('bad case: ' + brand + " " + newCat)
          #   if newCat.endswith('套装'):
          #     property = newCat[:-2]
          #     newCat = newCat[-2:]

          if (property == ""):
            newPredDebugSw.write(str(num) + "\n"
                                 + "http://item.jd.com/" + sku + ".html" + "\t" + cat3 + "\n"
                                 + brand + " " + newCat + "\t" + src + "\n"
                                 + writing_prediction + "\n\n")
            newPredSw.write("KB2Seq" + "\t" +
                            cat3 + "\t" +
                            kb['CategoryL3_id'][0] + "\t" +
                            kb['BrandName_cn'][0] + "\t" +
                            kb['Brand_id'][0].replace('\t', ' ') + "\t" +
                            sku + "\t" +
                            "http://item.jd.com/" + sku + ".html" + "\t" +
                            "\t" +
                            brand + " " + newCat + "\t" +
                            writing_prediction + "\t" +
                            str(writing_len) + "\n")
            if (len((brand + " " + newCat).replace(' ', '')) < 6 or len((brand + " " + newCat).replace(' ', '')) > 10):
              lengthError += 1
          else:
            newPredDebugSw.write(str(num) + "\n"
                                 + "http://item.jd.com/" + sku + ".html" + "\t" + cat3 + "\n"
                                 + brand + " " + property + " " + newCat + "\t" + src + "\n"
                                 + writing_prediction + "\n\n")
            newPredSw.write("KB2Seq" + "\t" +
                            cat3 + "\t" +
                            kb['CategoryL3_id'][0] + "\t" +
                            kb['BrandName_cn'][0] + "\t" +
                            kb['Brand_id'][0].replace('\t', ' ') + "\t" +
                            sku + "\t" +
                            "http://item.jd.com/" + sku + ".html" + "\t" +
                            "\t" +
                            brand + " " + property + " " + newCat + "\t" +
                            writing_prediction + "\t" +
                            str(writing_len) + "\n")
            if (len((brand + " " + property + " " + newCat).replace(' ', '')) < 6 or len(
                (brand + " " + property + " " + newCat).replace(' ', '')) > 10):
              lengthError += 1
          num += 1
  print('Length Error %d%d' % (lengthError, len(skus)))


# def rule_based_title_generation_forNewSku(skus, new_prediction_file, sku2info, sku2prediction, black_list,
#                                           bad_sku_list):
#   with open(new_prediction_file + ".forDebug", 'w', encoding='utf-8') as newPredDebugSw:
#     with open(new_prediction_file, 'w', encoding='utf-8') as newPredSw:
#       num = 1
#       lengthError = 0
#       for sku in skus:
#         old_title_prediction = sku2prediction[sku][0]
#         tokenized_writing = sku2prediction[sku][1].split(' ')
#         writing_prediction = sku2prediction[sku][1].replace(' ', '')
#         short_sentences = re.split('[，。]', writing_prediction)
#         short_sentences = [_ for _ in short_sentences if _ != '']
#         short_sentences_len = len(short_sentences)
#         writing_len = len(writing_prediction)
#
#         old_title_prediction_tokens = old_title_prediction.split(' ')
#
#         if sku not in sku2info:
#           if len(old_title_prediction_tokens) <= 3:
#             newPredSw.write(old_title_prediction)
#             print('?????????')
#           else:
#             newPredSw.write(
#               old_title_prediction_tokens[0] + " " + ''.join(old_title_prediction_tokens[1:-1]) + " " +
#               old_title_prediction_tokens[-1] + '\n')
#             print('?????????')
#           continue
#
#         kb = sku2info[sku]
#         cat3 = kb['CategoryL3'][0].replace(' ', '').strip()
#         brand = kb['BrandName_cn'][0].replace(' ', '').strip()
#         brandAll = kb['BrandName_full'][0].replace(' ', '').strip()
#
#         if cat3 == '':
#           print(sku)
#         if brand == '':
#           print('noBrand:', sku)
#         # if (cat3 in ['冰箱', '净水器', '空调', '洗衣机', '油烟机']):
#         if (True):
#           specialRecomandation = kb.get('特色推荐', [])
#
#           if (cat3 == '油烟机'):
#             specialRecomandation = kb.get('特色推荐', [])
#           if (cat3 == '洗衣机'):
#             specialRecomandation = kb.get('特色推荐', [])
#             if (specialRecomandation == ''):
#               tmp = [k.split('_')[1] for k, v in kb.items() if k.startswith('特色功能') and v == '支持']
#               if (len([tmp]) > 0):
#                 specialRecomandation = tmp
#             if (('变频' in writing_prediction or '变频' in old_title_prediction) and '主体_电机类型' in kb and '定频' in
#                 kb['主体_电机类型'][0]):
#               writing_prediction = writing_prediction.replace('变频', '定频')
#               old_title_prediction = old_title_prediction.replace('变频', '定频')
#               old_title_prediction_tokens = old_title_prediction.split(' ')
#               print('变=》定')
#             elif (('定频' in writing_prediction or '定频' in old_title_prediction) and '主体_电机类型' in kb and '变频' in
#                   kb['主体_电机类型'][0]):
#               writing_prediction = writing_prediction.replace('定频', '变频')
#               old_title_prediction = old_title_prediction.replace('定频', '变频')
#               old_title_prediction_tokens = old_title_prediction.split(' ')
#               print('定=》变')
#           if (cat3 == '冰箱'):
#             specialRecomandation = kb.get('特色推荐', [])
#             if (('变频' in writing_prediction or '变频' in old_title_prediction) and '压缩机' in kb and '定频' in kb['压缩机'][0]):
#               writing_prediction = writing_prediction.replace('变频', '定频')
#               old_title_prediction = old_title_prediction.replace('变频', '定频')
#               old_title_prediction_tokens = old_title_prediction.split(' ')
#               print('变=》定')
#             elif (
#                 ('定频' in writing_prediction or '定频' in old_title_prediction) and '压缩机' in kb and '变频' in kb['压缩机'][0]):
#               writing_prediction = writing_prediction.replace('定频', '变频')
#               old_title_prediction = old_title_prediction.replace('定频', '变频')
#               old_title_prediction_tokens = old_title_prediction.split(' ')
#               print('定=》变')
#             if (('风冷' in writing_prediction or '风冷' in old_title_prediction) and '制冷方式' in kb and (
#                 '风直冷' in kb['制冷方式'][0] or '混冷' in kb['制冷方式'][0])):
#               writing_prediction = writing_prediction.replace('风冷', '混冷')
#               old_title_prediction = old_title_prediction.replace('风冷', '混冷')
#               old_title_prediction_tokens = old_title_prediction.split(' ')
#               print('风=》混')
#             elif (('风冷' in writing_prediction or '风冷' in old_title_prediction) and '功能_制冷方式' in kb and (
#                 '风直冷' in kb['功能_制冷方式'][0] or '混冷' in kb['功能_制冷方式'][0])):
#               writing_prediction = writing_prediction.replace('风冷', '混冷')
#               old_title_prediction = old_title_prediction.replace('风冷', '混冷')
#               old_title_prediction_tokens = old_title_prediction.split(' ')
#               print('风=》混')
#           if (cat3 == '净水器'):
#             specialRecomandation = kb.get('特色推荐', [])
#             if (specialRecomandation == ''):
#               specialRecomandation = kb.get('产品特点', [])
#           if (cat3 == '空调'):
#             specialRecomandation = kb.get('产品特点', [])
#             if (('变频' in writing_prediction or '变频' in old_title_prediction) and '变频/定频' in kb and '定频' in kb['变频/定频'][
#               0]):
#               writing_prediction = writing_prediction.replace('变频', '定频')
#               old_title_prediction = old_title_prediction.replace('变频', '定频')
#               old_title_prediction_tokens = old_title_prediction.split(' ')
#               print('变=》定')
#             elif (('变频' in writing_prediction or '变频' in old_title_prediction) and '功能_变频/定频' in kb and '定频' in
#                   kb['功能_变频/定频'][0]):
#               writing_prediction = writing_prediction.replace('变频', '定频')
#               old_title_prediction = old_title_prediction.replace('变频', '定频')
#               old_title_prediction_tokens = old_title_prediction.split(' ')
#               print('变=》定')
#             elif (
#                 ('定频' in writing_prediction or '定频' in old_title_prediction) and '变频/定频' in kb and '变频' in kb['变频/定频'][
#               0]):
#               writing_prediction = writing_prediction.replace('定频', '变频')
#               old_title_prediction = old_title_prediction.replace('定频', '变频')
#               old_title_prediction_tokens = old_title_prediction.split(' ')
#               print('定=》变')
#             elif (('定频' in writing_prediction or '定频' in old_title_prediction) and '功能_变频/定频' in kb and '变频' in
#                   kb['功能_变频/定频'][0]):
#               writing_prediction = writing_prediction.replace('定频', '变频')
#               old_title_prediction = old_title_prediction.replace('定频', '变频')
#               old_title_prediction_tokens = old_title_prediction.split(' ')
#               print('定=》变')
#
#           propertyToks = []
#           idx = 0
#           while True:
#             if (idx >= len(old_title_prediction_tokens)):
#               break
#             if (old_title_prediction_tokens[idx] not in brandAll):
#               break
#             idx += 1
#
#           while True:
#             if (idx >= len(old_title_prediction_tokens)):
#               break
#             tmpProperty = ''.join(propertyToks)
#             if tmpProperty.startswith(brand.lower()):
#               tmpProperty = tmpProperty[len(brand):]
#             if (len(set(old_title_prediction_tokens[idx]) & set(cat3)) > 0 and len(tmpProperty) > 1):
#               break
#             propertyToks.append(old_title_prediction_tokens[idx])
#             idx += 1
#
#           src = "None"
#
#           property = ''
#           if (True):
#             if property == '' and len(propertyToks) > 0:
#               property = ''.join(propertyToks)
#               src = "FromS2S"
#
#             # tmp = re.sub(pattern=pat, repl='啊', string=title.replace(' ', ''))
#             # while (len(tmp) < 6 or len(tmp) > 10):
#
#             if (property == '' or (re.match(pattern, property) != None)) and len(specialRecomandation) > 0:
#               property = specialRecomandation[0]
#               src = "FromKB"
#           if (False):
#             if property == '' and len(specialRecomandation) > 0:
#               property = specialRecomandation[0]
#               src = "FromKB"
#             if property == '' and len(propertyToks) > 0:
#               property = ''.join(propertyToks)
#               src = "FromS2S"
#
#           newCat = ''.join(old_title_prediction_tokens[idx:])
#           if property.startswith(brand.lower()):
#             property = property[len(brand):]
#           if property.endswith(cat3.lower()):
#             property = property[:len(property) - len(cat3)]
#           if property.endswith(newCat.lower()):
#             property = property[:len(property) - len(newCat)]
#
#           # if (property == ""):
#           #   print('bad case: ' + brand + " " + newCat)
#           #   if newCat.endswith('套装'):
#           #     property = newCat[:-2]
#           #     newCat = newCat[-2:]
#
#           if (property == ""):
#             newPredDebugSw.write(str(num) + "\n"
#                                  + "http://item.jd.com/" + sku + ".html" + "\t" + cat3 + "\n"
#                                  + brand + " " + newCat + "\t" + src + "\n"
#                                  + writing_prediction + "\n\n")
#             newPredSw.write("KB2Seq" + "\t" +
#                             cat3 + "\t" +
#                             kb['CategoryL3_id'][0] + "\t" +
#                             kb['BrandName_cn'][0] + "\t" +
#                             kb['Brand_id'][0].replace('\t', ' ') + "\t" +
#                             sku + "\t" +
#                             "http://item.jd.com/" + sku + ".html" + "\t" +
#                             "\t" +
#                             brand + " " + newCat + "\t" +
#                             writing_prediction + "\t" +
#                             str(writing_len) + "\n")
#             if (len((brand + " " + newCat).replace(' ', '')) < 6 or len((brand + " " + newCat).replace(' ', '')) > 10):
#               lengthError += 1
#           else:
#             newPredDebugSw.write(str(num) + "\n"
#                                  + "http://item.jd.com/" + sku + ".html" + "\t" + cat3 + "\n"
#                                  + brand + " " + property + " " + newCat + "\t" + src + "\n"
#                                  + writing_prediction + "\n\n")
#             newPredSw.write("KB2Seq" + "\t" +
#                             cat3 + "\t" +
#                             kb['CategoryL3_id'][0] + "\t" +
#                             kb['BrandName_cn'][0] + "\t" +
#                             kb['Brand_id'][0].replace('\t', ' ') + "\t" +
#                             sku + "\t" +
#                             "http://item.jd.com/" + sku + ".html" + "\t" +
#                             "\t" +
#                             brand + " " + property + " " + newCat + "\t" +
#                             writing_prediction + "\t" +
#                             str(writing_len) + "\n")
#             if (len((brand + " " + property + " " + newCat).replace(' ', '')) < 6 or len(
#                 (brand + " " + property + " " + newCat).replace(' ', '')) > 10):
#               lengthError += 1
#           num += 1
#   print('Length Error %d%d' % (lengthError, len(skus)))

def rule_based_title_generation_forNewSku(skus, new_prediction_file, sku2info, sku2prediction, black_list,
                                          bad_sku_list, model_version, brandName2aliases):
  with open(new_prediction_file + ".forDebug", 'w', encoding='utf-8') as newPredDebugSw:
    with open(new_prediction_file, 'w', encoding='utf-8') as newPredSw:
      num = 1
      lengthError = 0
      for sku in skus:
        old_title_prediction = sku2prediction[sku][0]
        tokenized_writing = sku2prediction[sku][1].split(' ')
        writing_prediction = sku2prediction[sku][1].replace(' ', '')
        short_sentences = re.split('[，。]', writing_prediction)
        short_sentences = [_ for _ in short_sentences if _ != '']
        short_sentences_len = len(short_sentences)
        writing_len = len(writing_prediction)

        old_title_prediction_tokens = old_title_prediction.split(' ')

        if sku not in sku2info:
          if len(old_title_prediction_tokens) <= 3:
            newPredSw.write(old_title_prediction)
            print('?????????')
          else:
            newPredSw.write(
              old_title_prediction_tokens[0] + " " + ''.join(old_title_prediction_tokens[1:-1]) + " " +
              old_title_prediction_tokens[-1] + '\n')
            print('?????????')
          continue

        kb = sku2info[sku]
        cat3 = kb['CategoryL3'][0].replace(' ', '').strip()
        brand = kb['BrandName_cn'][0].replace(' ', '').strip()
        brandEn = kb['BrandName_en'][0].replace(' ', '').strip()
        brandAll = kb['BrandName_full'][0].replace(' ', '').strip()
        brandAliases = []
        if brand in brandName2aliases:
          brandAliases = brandName2aliases[brand]
        elif brandEn in brandName2aliases:
          brandAliases = brandName2aliases[brand]

        if cat3 == '':
          print(sku)
        if brand == '':
          print('noBrand:', sku)
        # if (cat3 in ['冰箱', '净水器', '空调', '洗衣机', '油烟机']):
        if (True):
          specialRecomandation=[]
          specialRecomandation = kb.get('特色推荐', [])
          specialRecomandation.extend(kb.get('产品特点', []))
          specialRecomandation.extend(kb.get('产品特色', []))
          tmp = [k.split('_')[1] for k, v in kb.items() if k.startswith('特色功能') and v == '支持']
          if (len([tmp]) > 0):
            specialRecomandation.extend(tmp)

          # if (cat3 == '油烟机'):
          #   specialRecomandation = specialRecomandation
          if (cat3 == '洗衣机'):
            # if (len(specialRecomandation) == 0):
            #   tmp = [k.split('_')[1] for k, v in kb.items() if k.startswith('特色功能') and v == '支持']
            #   if (len([tmp]) > 0):
            #     specialRecomandation = tmp
            if (('变频' in writing_prediction or '变频' in old_title_prediction) and '主体_电机类型' in kb and '定频' in kb['主体_电机类型'][0]):
              writing_prediction = writing_prediction.replace('变频', '定频')
              old_title_prediction = old_title_prediction.replace('变频', '定频')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('变=》定')
            elif (('定频' in writing_prediction or '定频' in old_title_prediction) and '主体_电机类型' in kb and '变频' in kb['主体_电机类型'][0]):
              writing_prediction = writing_prediction.replace('定频', '变频')
              old_title_prediction = old_title_prediction.replace('定频', '变频')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('定=》变')
          if (cat3 == '冰箱'):
            if (('变频' in writing_prediction or '变频' in old_title_prediction) and '压缩机' in kb and '定频' in kb['压缩机'][0]):
              writing_prediction = writing_prediction.replace('变频', '定频')
              old_title_prediction = old_title_prediction.replace('变频', '定频')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('变=》定')
            elif (('定频' in writing_prediction or '定频' in old_title_prediction) and '压缩机' in kb and '变频' in kb['压缩机'][0]):
              writing_prediction = writing_prediction.replace('定频', '变频')
              old_title_prediction = old_title_prediction.replace('定频', '变频')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('定=》变')
            if (('风冷' in writing_prediction or '风冷' in old_title_prediction) and '制冷方式' in kb and ('风直冷' in kb['制冷方式'][0] or '混冷' in kb['制冷方式'][0])):
              writing_prediction = writing_prediction.replace('风冷', '混冷')
              old_title_prediction = old_title_prediction.replace('风冷', '混冷')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('风=》混')
            elif (('风冷' in writing_prediction or '风冷' in old_title_prediction) and '功能_制冷方式' in kb and ('风直冷' in kb['功能_制冷方式'][0] or '混冷' in kb['功能_制冷方式'][0])):
              writing_prediction = writing_prediction.replace('风冷', '混冷')
              old_title_prediction = old_title_prediction.replace('风冷', '混冷')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('风=》混')
          # if (cat3 == '净水器'):
          #   specialRecomandation = specialRecomandation
          if (cat3 == '空调'):
            if (('变频' in writing_prediction or '变频' in old_title_prediction) and '变频/定频' in kb and '定频' in kb['变频/定频'][0]):
              writing_prediction = writing_prediction.replace('变频', '定频')
              old_title_prediction = old_title_prediction.replace('变频', '定频')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('变=》定')
            elif (('变频' in writing_prediction or '变频' in old_title_prediction) and '功能_变频/定频' in kb and '定频' in kb['功能_变频/定频'][0]):
              writing_prediction = writing_prediction.replace('变频', '定频')
              old_title_prediction = old_title_prediction.replace('变频', '定频')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('变=》定')
            elif (('定频' in writing_prediction or '定频' in old_title_prediction) and '变频/定频' in kb and '变频' in kb['变频/定频'][0]):
              writing_prediction = writing_prediction.replace('定频', '变频')
              old_title_prediction = old_title_prediction.replace('定频', '变频')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('定=》变')
            elif (('定频' in writing_prediction or '定频' in old_title_prediction) and '功能_变频/定频' in kb and '变频' in kb['功能_变频/定频'][0]):
              writing_prediction = writing_prediction.replace('定频', '变频')
              old_title_prediction = old_title_prediction.replace('定频', '变频')
              old_title_prediction_tokens = old_title_prediction.split(' ')
              print('定=》变')

          propertyToks = []
          idx = 0
          while True: #find token index after brand
            if (idx >= len(old_title_prediction_tokens)):
              break
            # if (old_title_prediction_tokens[idx] not in brandAll):
            #   break
            if (all([old_title_prediction_tokens[idx].lower() not in alias.lower() for alias in brandAliases])):
              break
            idx += 1

          while True:#find selling point (middle) as $propertyToks
            if (idx >= len(old_title_prediction_tokens)):
              break
            tmpProperty = ''.join(propertyToks)
            if tmpProperty.startswith(brand.lower()):
              tmpProperty = tmpProperty[len(brand):]
            if (len(set(old_title_prediction_tokens[idx]) & set(cat3)) > 0 and len(tmpProperty) > 1):
              break
            propertyToks.append(old_title_prediction_tokens[idx])
            idx += 1

          src = "None"

          property = ''
          if (True):
            if property == '' and len(propertyToks) > 0:
              property = ''.join(propertyToks)
              #for bra in brandAliases:
              #  property = property.replace(bra, '')
              property = property.replace(brand, '')
              if isCountry(property):
                property = ''
              src = "FromS2S"

            # tmp = re.sub(pattern=pat, repl='啊', string=title.replace(' ', ''))
            # while (len(tmp) < 6 or len(tmp) > 10):

            if (property == '' or (re.match(pattern, property) != None)) and len(specialRecomandation) > 0:
              property = specialRecomandation[0]
              src = "FromKB"
          if (False):
            if property == '' and len(specialRecomandation) > 0:
              property = specialRecomandation[0]
              src = "FromKB"
            if property == '' and len(propertyToks) > 0:
              property = ''.join(propertyToks)
              src = "FromS2S"

          newCat = ''.join(old_title_prediction_tokens[idx:])
          if(newCat == ''):
            newCat = cat3

          tmp_title = brand + " " + property + " " + newCat
          if float(titleLength(tmp_title)) > 10 and len(specialRecomandation) > 0:
            sortRec = sorted(specialRecomandation, key=lambda x: len(x), reverse=False)
            property = sortRec[0]
            tmp_title = brand + " " + property + " " + newCat
          if float(titleLength(tmp_title)) > 10:
            newCat = newCat.replace('空气净化器', '净化器')
            tmp_title = brand + " " + property + " " + newCat
          if float(titleLength(tmp_title)) > 10 and len(property) > 0:
            property = property.replace('定点按摩', '定点')
            tmp_title = brand + " " + property + " " + newCat
            
          if property.startswith(brand.lower()):
            property = property[len(brand):]
          if property.endswith(cat3.lower()):
            property = property[:len(property) - len(cat3)]
          if property.endswith(newCat.lower()):
            property = property[:len(property) - len(newCat)]
          # if (property == ""):
          #   print('bad case: ' + brand + " " + newCat)
          #   if newCat.endswith('套装'):
          #     property = newCat[:-2]
          #     newCat = newCat[-2:]

          if (property == ""):
            newPredDebugSw.write(str(num) + "\n"
                                 + "http://item.jd.com/" + sku + ".html" + "\t" + cat3 + "\n"
                                 + brand + " " + newCat + "\t" + src + "\n"
                                 + writing_prediction + "\n\n")
            newPredSw.write("" + "\t" +
                            cat3 + "\t" +
                            kb['CategoryL3_id'][0] + "\t" +
                            kb['BrandName_cn'][0] + "\t" +
                            kb['Brand_id'][0].replace('\t', ' ') + "\t" +
                            sku + "\t" +
                            "http://item.jd.com/" + sku + ".html" + "\t" +
                            model_version + "\t\t\t\t\t\t\t" +
                            brand + " " + newCat + "\t\t\t\t\t\t" +
                            writing_prediction + "\t\t\t\t\t\t\t\t\t\n")
            # if (len((brand + " " + newCat).replace(' ', '')) < 6 or len((brand + " " + newCat).replace(' ', '')) > 10):
            #   lengthError += 1
          else:
            newPredDebugSw.write(str(num) + "\n"
                                 + "http://item.jd.com/" + sku + ".html" + "\t" + cat3 + "\n"
                                 + brand + " " + property + " " + newCat + "\t" + src + "\n"
                                 + writing_prediction + "\n\n")
            newPredSw.write("" + "\t" +
                            cat3 + "\t" +
                            kb['CategoryL3_id'][0] + "\t" +
                            kb['BrandName_cn'][0] + "\t" +
                            kb['Brand_id'][0].replace('\t', ' ') + "\t" +
                            sku + "\t" +
                            "http://item.jd.com/" + sku + ".html" + "\t" +
                            model_version + "\t\t\t\t\t\t\t" +
                            brand + " " + property + " " + newCat + "\t\t\t\t\t\t" +
                            writing_prediction + "\t\t\t\t\t\t\t\t\t\n")
            # if (len((brand + " " + property + " " + newCat).replace(' ', '')) < 6 or len(
            #     (brand + " " + property + " " + newCat).replace(' ', '')) > 10):
            #   lengthError += 1
          num += 1
  print('Length Error %d%d' % (lengthError, len(skus)))

def rule_based_title_generation(test_sku_file, new_prediction_file, sku2info, sku2prediction):
  with open(test_sku_file, 'r', encoding='utf-8') as skusSr:
    with open(new_prediction_file, 'w', encoding='utf-8') as newPredSw:
      with open(new_prediction_file + ".达人", 'w', encoding='utf-8') as humanSw:
        skus = [_.strip() for _ in skusSr]

        num = 1
        for sku in skus:
          old_title_prediction = sku2prediction[sku][0]
          writing_prediction = sku2prediction[sku][1].replace(' ', '')

          old_title_prediction_tokens = old_title_prediction.split(' ')
          if sku not in sku2info:
            if len(old_title_prediction_tokens) <= 3:
              newPredSw.write(old_title_prediction)
            else:
              newPredSw.write(
                old_title_prediction_tokens[0] + " " + ''.join(old_title_prediction_tokens[1:-1]) + " " +
                old_title_prediction_tokens[-1] + '\n')
            continue
          info = sku2info[sku]
          kb = info[0][1]
          text = info[0][3]
          cat3 = kb['CategoryL3'][0].replace(' ', '').strip()
          brand = kb['BrandName_cn'][0].replace(' ', '').strip()
          brandAll = kb['BrandName_full'][0].replace(' ', '').strip()

          if cat3 == '':
            print(sku)
          if brand == '':
            print('noBrand:', sku)
          if (cat3 in ['冰箱', '净水器', '空调', '洗衣机', '油烟机']):

            specialRecomandation = []
            if (cat3 == '油烟机'):
              specialRecomandation = kb.get('特色推荐', [])

            if (cat3 == '洗衣机'):
              specialRecomandation = kb.get('特色推荐', [])
              if (specialRecomandation == ''):
                tmp = [k.split('_')[1] for k, v in kb.items() if k.startswith('特色功能') and v == '支持']
                if (len([tmp]) > 0):
                  specialRecomandation = tmp

            if (cat3 == '冰箱'):
              specialRecomandation = kb.get('特色推荐', [])

            if (cat3 == '净水器'):
              specialRecomandation = kb.get('特色推荐', [])
              if (specialRecomandation == ''):
                specialRecomandation = kb.get('产品特点', [])

            if (cat3 == '空调'):
              specialRecomandation = kb.get('产品特点', [])

            propertyToks = []
            idx = 0
            while True:
              if (idx >= len(old_title_prediction_tokens)):
                break
              if (old_title_prediction_tokens[idx] not in brandAll):
                break
              idx += 1

            while True:
              if (idx >= len(old_title_prediction_tokens)):
                break
              if (len(set(old_title_prediction_tokens[idx]) & set(cat3)) > 0):
                break
              propertyToks.append(old_title_prediction_tokens[idx])
              idx += 1

            src = "None"

            property = ''
            if (True):
              if property == '' and len(propertyToks) > 0:
                property = ''.join(propertyToks)
                src = "FromS2S"
              if property == '' and len(specialRecomandation) > 0:
                property = specialRecomandation[0]
                src = "FromKB"
            if (False):
              if property == '' and len(specialRecomandation) > 0:
                property = specialRecomandation[0]
                src = "FromKB"
              if property == '' and len(propertyToks) > 0:
                property = ''.join(propertyToks)
                src = "FromS2S"

            newCat = ''.join(old_title_prediction_tokens[idx:])
            if property.startswith(brand.lower()):
              property = property[len(brand):]
            if property.endswith(cat3.lower()):
              property = property[:len(property) - len(cat3)]
            if property.endswith(newCat.lower()):
              property = property[:len(property) - len(newCat)]

            if (property == ""):
              print('bad case: ' + brand + " " + newCat)
              newPredSw.write(str(num) + "\n"
                              + "http://item.jd.com/" + sku + ".html" + "\t" + cat3 + "\n"
                              + brand + " " + newCat + "\t" + src + "\n"
                              + writing_prediction + "\n\n")
              humanSw.write(text + "\thttp://item.jd.com/" + sku + ".html" + "\n")
            else:
              newPredSw.write(str(num) + "\n"
                              + "http://item.jd.com/" + sku + ".html" + "\t" + cat3 + "\n"
                              + brand + " " + property + " " + newCat + "\t" + src + "\n"
                              + writing_prediction + "\n\n")
              humanSw.write(text + "\thttp://item.jd.com/" + sku + ".html" + "\n")

            num += 1


def loadSku2prediction(skus, title_prediction_file, writingPrediction_file):
  sku2prediction = {}
  with open(title_prediction_file, 'r', encoding='utf-8') as titlePredictionsSr:
    with open(writingPrediction_file, 'r', encoding='utf-8') as writingPredictionsSr:
      titles = [_.strip('\n 。') for _ in titlePredictionsSr.readlines()]
      writings = [_.strip() for _ in writingPredictionsSr.readlines()]

      i = 0
      for sku in skus:
        sku2prediction[sku] = (titles[i], writings[i])
        i += 1
  return sku2prediction


def loadSku2SellPoints(skus, sellPoints_File):
  sps = [_.strip().split('\t') for _ in open(sellPoints_File, 'r', encoding='utf-8')]
  sku2sps = {}
  index = 0
  for sku in skus:
    sku2sps[sku] = sps[index]
    index += 1
  return sku2sps


def get_input_info_for_new_sku(new_sku_file, skus, skuBasicInfo):
  with open(new_sku_file + '.basicInfo', 'w', encoding='utf-8') as f_out_basicInfo:
    cnt = 0
    for sku in skus:
      if sku in skuBasicInfo:
        f_out_basicInfo.write("%s\t%s\t%s\t%s\t%s\n"
                              % (skuBasicInfo[sku]['CategoryL3'][0],
                                 skuBasicInfo[sku]['CategoryL3_id'][0],
                                 skuBasicInfo[sku]['BrandName_cn'][0],
                                 skuBasicInfo[sku]['Brand_id'][0],
                                 sku))
      else:
        cnt += 1
        f_out_basicInfo.write("%s\t%s\t%s\t%s\t%s\n"
                              % ('',
                                 '',
                                 '',
                                 '',
                                 sku))
    print('%s NOT found!!!' % cnt)


def filtData(final_file, validsku2pics, sku2sps, bad_sku_list):
  # pat = r'[^\u4e00-\u9fa5]+'
  pat = r'[\u4e00-\u9fa5]'

  sku2result = {}
  cnt = 0
  bad_tokens = ['()', '+', '-']
  label = True
  reason = 'Pass!'
  skus = set()
  good = 0
  print('=====bad_sku_list======%d===========' % len(bad_sku_list))
  with open(final_file, 'r', encoding='utf-8') as f:
    while (True):
      line = f.readline()
      reason = 'Pass!'
      label = True
      if (not line):
        break
      line = line.strip()

      parts = line.split('\t')
      sku = parts[5].strip()
      skus.add(sku)
      writing_prediction = parts[9].strip()
      if writing_prediction.endswith('，') or writing_prediction.endswith('；'):
        writing_prediction = writing_prediction[:-1] + '。'
      if not any([writing_prediction.endswith(punc) for punc in ['。', '！', '？']]):
        writing_prediction = writing_prediction + '。'
      parts[9] = writing_prediction

      # if ('过滤精度' in line):
      #   jingdu_pat = '[，。；！、]([^，。；！]*过滤精度[^，。；！、]*)[，。；！、]'
      #   m = re.search(jingdu_pat, writing_prediction)
      #   writing_prediction = writing_prediction.replace(m.groups()[0], '过滤精度高')
      #   parts[9] = writing_prediction
      #   line = '\t'.join(parts)
      #   # label = False
      #   # reason = '过滤精度'
      #   sku2result[sku] = (line, label, reason)

      if (sku in bad_sku_list):
        label = False
        reason = '审核不通过'
        line = '\t'.join(parts)
        sku2result[sku] = (line, label, reason)
        continue

      title = parts[8].strip()
      tmp = re.sub(pattern=pat, repl='aa', string=title.replace(' ', ''))
      # print(tmp)
      ps3 = title.split(' ')

      if (len(tmp) > 20 and title.endswith('洗衣机') and len(ps3[-1]) > len('洗衣机')):
        new_title = ps3[0] + ' ' + ps3[-1][:-1*len('洗衣机')] + ' ' + '洗衣机'
        parts[8] = new_title
        line = '\t'.join(parts)
      title = parts[8].strip()
      tmp = re.sub(pattern=pat, repl='aa', string=title.replace(' ', ''))

      if (len(tmp) < 12 or len(tmp) > 20):
        cnt += 1
        print("length: %s" % title)
        label = False
        reason = 'title长度不符合'
        line = '\t'.join(parts)
        sku2result[sku] = (line, label, reason)
        continue
      if (len(title.split(' ')) != 3):
        prodWords = ['挂机', '柜机', '管线机', '热水器', '四件套', '三件套', '直饮机', '空调']
        flag = False
        for prodW in prodWords:
          if (title.endswith(prodW) and len(title.split(' ')[-1]) > len(prodW)):
            new_title = title.replace(prodW, ' ' + prodW)
            parts[8] = new_title
            line = '\t'.join(parts)
            flag = True
            break
        if (not flag):
          cnt += 1
          print('len(title.split(' ')) != 3: %s' % title)
          label = False
          reason = 'title不是3段'
          sku2result[sku] = (line, label, reason)
          line = '\t'.join(parts)
          continue

      writing_prediction = parts[9].strip()
      short_sentences = re.split('[，。；！]', writing_prediction)
      short_sentences = [_ for _ in short_sentences if _ != '']
      short_sentences_len = len(short_sentences)
      if (short_sentences_len <= 2 or short_sentences_len >= 16 or any([len(sent)>30 for sent in short_sentences])):
        label = False
        reason = '短文：短句数太长或太短'
        line = '\t'.join(parts)
        sku2result[sku] = (line, label, reason)
        continue
      bad_flag = False
      for bad_word in bad_tokens:
        if (bad_word in writing_prediction):
          bad_flag = True
          break
      if (bad_flag):
        label = False
        reason = '短文：包含违禁词'
        line = '\t'.join(parts)
        sku2result[sku] = (line, label, reason)
        continue
      line = '\t'.join(parts)
      sku2result[sku] = (line, label, reason)
      good += 1

  print('too long or short %s' % cnt)
  print('good sku: %d' % good)
  print('==========%d========' % len(list(skus)))

  with open(final_file + ".valid", 'w', encoding='utf-8') as sw:
    id = 0
    for sku, pics in validsku2pics.items():
      id += 1
      if sku in sku2result:
        line, label, reason = sku2result[sku]
        sps = sku2sps[sku]
        # parts = line.split('\t')
        # title = parts[8]
        # if (len(title.replace(' ', '')) < 6 or len(title.replace(' ', '')) > 10):
        #   cnt += 1
        #   print(title)
        # if (len(title.split(' ')) != 3):
        #   cnt += 1
        #   print(title)
        sw.write(str(label) + '\t' + reason + '\t' +  str(id) + '\t' + line + '\t' + str(sps) + '\t' + '\t'.join(pics) + '\n')
      else:
        reason = '未获取sku对应KB信息'
        sw.write(
          'False\t' + reason + '\t' + str(
            id) + '\tKB2Seq\tNULL\tNULL\tNULL\tNULL\t' + sku + '\thttp://item.jd.com/' + sku + '.html\t\tFake\tFake\t0\n')
        print(sku)

def filtData_allDoamin(final_file, validsku2pics, sku2sps, bad_sku_list, model_version):
  # pat = r'[^\u4e00-\u9fa5]+'
  pat = r'[\u4e00-\u9fa5]'

  sku2result = {}
  cnt = 0
  bad_tokens = ['()', '+', '-']
  label = True
  reason = 'Pass!'
  skus = set()
  good = 0
  print('=====bad_sku_list======%d===========' % len(bad_sku_list))
  with open(final_file, 'r', encoding='utf-8') as f:
    while (True):
      line = f.readline()
      reason = 'Pass!'
      label = True
      if (not line):
        break
      line = line.strip('\n')

      parts = line.split('\t')
      sku = parts[5].strip()
      skus.add(sku)
      writing_prediction = parts[20].strip()
      if writing_prediction.endswith('，') or writing_prediction.endswith('；'):
        writing_prediction = writing_prediction[:-1] + '。'
      if not any([writing_prediction.endswith(punc) for punc in ['。', '！', '？']]):
        writing_prediction = writing_prediction + '。'
      parts[20] = writing_prediction
      parts[21] = str(len(writing_prediction))

      # if ('过滤精度' in line):
      #   jingdu_pat = '[，。；！、]([^，。；！]*过滤精度[^，。；！、]*)[，。；！、]'
      #   m = re.search(jingdu_pat, writing_prediction)
      #   writing_prediction = writing_prediction.replace(m.groups()[0], '过滤精度高')
      #   parts[9] = writing_prediction
      #   line = '\t'.join(parts)
      #   # label = False
      #   # reason = '过滤精度'
      #   sku2result[sku] = (line, label, reason)

      if (sku in bad_sku_list):
        label = False
        reason = '审核不通过'
        line = '\t'.join(parts)
        sku2result[sku] = (line, label, reason)
        continue

      title = parts[14].strip()
      tmp = re.sub(pattern=pat, repl='aa', string=title.replace(' ', ''))
      # print(tmp)
      ps3 = title.split(' ')

      if (len(tmp) > 20 and title.endswith('洗衣机') and len(ps3[-1]) > len('洗衣机')):
        new_title = ps3[0] + ' ' + ps3[-1][:-1*len('洗衣机')] + ' ' + '洗衣机'
        parts[14] = new_title
        line = '\t'.join(parts)
      title = parts[14].strip()
      tmp = re.sub(pattern=pat, repl='aa', string=title.replace(' ', ''))

      if (len(tmp) < 12 or len(tmp) > 20):
        cnt += 1
        print("length: %s" % title)
        label = False
        reason = 'title长度不符合'
        line = '\t'.join(parts)
        sku2result[sku] = (line, label, reason)
        continue

      if (len(title.split(' ')) != 3):
        prodWords = ['挂机', '柜机', '管线机', '热水器', '四件套', '三件套', '直饮机', '空调']
        flag = False
        for prodW in prodWords:
          if (title.endswith(prodW) and len(title.split(' ')[-1]) > len(prodW)):
            new_title = title.replace(prodW, ' ' + prodW)
            parts[14] = new_title
            line = '\t'.join(parts)
            flag = True
            break
        if (not flag):
          cnt += 1
          print('len(title.split(' ')) != 3: %s' % title)
          label = False
          reason = 'title不是3段'
          sku2result[sku] = (line, label, reason)
          line = '\t'.join(parts)
          continue
      title = parts[14].strip()
      tmp = re.sub(pattern=pat, repl='aa', string=title.replace(' ', ''))
      parts[15] = str(len(tmp) / 2)
      writing_prediction = parts[20].strip()
      short_sentences = re.split('[，。；！]', writing_prediction)
      short_sentences = [_ for _ in short_sentences if _ != '']
      short_sentences_len = len(short_sentences)
      if (short_sentences_len <= 2 or short_sentences_len >= 16 or any([len(sent)>30 for sent in short_sentences])):
        label = False
        reason = '短文：短句数太长或太短'
        line = '\t'.join(parts)
        sku2result[sku] = (line, label, reason)
        continue
      bad_flag = False
      for bad_word in bad_tokens:
        if (bad_word in writing_prediction):
          bad_flag = True
          break
      if (bad_flag):
        label = False
        reason = '短文：包含违禁词'
        line = '\t'.join(parts)
        sku2result[sku] = (line, label, reason)
        continue
      line = '\t'.join(parts)
      sku2result[sku] = (line, label, reason)
      good += 1

  print('too long or short %s' % cnt)
  print('good sku: %d' % good)
  print('==========%d========' % len(list(skus)))

  with open(final_file + ".valid", 'w', encoding='utf-8') as sw:
    id = 0
    for sku, pics in validsku2pics.items():
      id += 1
      sps = []
      if sku in sku2result:
        line, label, reason = sku2result[sku]
        sps = sku2sps[sku]
        # parts = line.split('\t')
        # title = parts[8]
        # if (len(title.replace(' ', '')) < 6 or len(title.replace(' ', '')) > 10):
        #   cnt += 1
        #   print(title)
        # if (len(title.split(' ')) != 3):
        #   cnt += 1
        #   print(title)
        sw.write(str(label) + '\t' + reason + '\t' + str(sps) + '\t' +  str(id) + line + '\t' + '\t'.join(pics) + '\n')
      else:
        reason = '未获取sku对应KB信息'
        sw.write(
          'False\t' + reason + '\t'  + str(sps) + '\t' + str(id) + '\tNULL\tNULL\tNULL\tNULL\t' + sku + '\thttp://item.jd.com/' + sku + '.html\t' +
          model_version + '\t\t\t\t\t\tFake\tFake\t\t\t\tFake\tFake\t\t\t\t\t\t\t\t\n')
        print(sku)

def filtData_allDoamin_joinPic(final_file, validsku2pics, sku2sps, bad_sku_list, model_version):
  # pat = r'[^\u4e00-\u9fa5]+'
  pat = r'[\u4e00-\u9fa5]'

  sku2result = {}
  cnt = 0
  bad_tokens = ['()', '+', '-']
  label = True
  reason = 'Pass!'
  skus = set()
  good = 0
  print('=====bad_sku_list======%d===========' % len(bad_sku_list))
  with open(final_file, 'r', encoding='utf-8') as f:
    while (True):
      line = f.readline()
      reason = 'Pass!'
      label = True
      if (not line):
        break
      line = line.strip('\n')

      parts = line.split('\t')
      sku = parts[5].strip()
      skus.add(sku)
      writing_prediction = parts[20].strip()
      if writing_prediction.endswith('，') or writing_prediction.endswith('；'):
        writing_prediction = writing_prediction[:-1] + '。'
      if not any([writing_prediction.endswith(punc) for punc in ['。', '！', '？']]):
        writing_prediction = writing_prediction + '。'
      parts[20] = writing_prediction
      parts[21] = str(len(writing_prediction))

      # if (sku in bad_sku_list):
      #   label = False
      #   reason = '审核不通过'
      #   line = '\t'.join(parts)
      #   sku2result[sku] = (line, label, reason)
      #   continue

      title = parts[14].strip().replace("【京东配送】","")
      parts[14] = title
      tmp = re.sub(pattern=pat, repl='aa', string=title.replace(' ', ''))
      # print(tmp)
      ps3 = title.split(' ')

      if (len(tmp) > 20 and title.endswith('洗衣机') and len(ps3[-1]) > len('洗衣机')):
        new_title = ps3[0] + ' ' + ps3[-1][:-1*len('洗衣机')] + ' ' + '洗衣机'
        parts[14] = new_title
        parts[15] = titleLength(parts[14])
        line = '\t'.join(parts)
      if (len(ps3) == 3):
        brand = ps3[0]
        mid = ps3[1]
        cat = ps3[2]
        mid = mid.replace('美的','')
        mid = mid.replace('格力', '')
        cat = cat.replace(brand, '')
        new_title = brand + ' ' + mid + ' ' + cat
        parts[14] = new_title
        parts[15] = titleLength(parts[14])
        line = '\t'.join(parts)


      # title = parts[14].strip()
      # tmp = re.sub(pattern=pat, repl='aa', string=title.replace(' ', ''))
      #
      # if (len(tmp) < 12 or len(tmp) > 20):
      #   cnt += 1
      #   print("length: %s" % title)
      #   label = False
      #   reason = 'title长度不符合'
      #   parts[15] = titleLength(title)
      #   line = '\t'.join(parts)
      #   sku2result[sku] = (line, label, reason)
      #   continue

      if (len(title.split(' ')) != 3):
        prodWords = ['挂机', '柜机', '管线机', '热水器', '四件套', '三件套', '直饮机', '空调']
        # flag = False
        for prodW in prodWords:
          if (title.endswith(prodW) and len(title.split(' ')[-1]) > len(prodW)):
            new_title = title.replace(prodW, ' ' + prodW)
            parts[14] = new_title
            parts[15] = titleLength(parts[14])
            line = '\t'.join(parts)
            # flag = True
            break

        # if (not flag):
        #   cnt += 1
        #   print('len(title.split(' ')) != 3: %s' % title)
        #   label = False
        #   reason = 'title不是3段'
        #   parts[15] = titleLength(title)
        #   line = '\t'.join(parts)
        #   sku2result[sku] = (line, label, reason)
        #   continue

      parts[15] = titleLength(parts[14])
      writing_prediction = parts[20].strip()
      short_sentences = re.split('[，。；！]', writing_prediction)
      short_sentences = [_ for _ in short_sentences if _ != '']
      short_sentences_len = len(short_sentences)
      if (short_sentences_len <= 2 or short_sentences_len >= 16 or any([len(sent)>30 for sent in short_sentences])):
        label = False
        reason = '短文：短句数太长或太短'
        line = '\t'.join(parts)
        sku2result[sku] = (line, label, reason)
        continue

      bad_flag = False
      for bad_word in bad_tokens:
        if (bad_word in writing_prediction):
          bad_flag = True
          break
      if (bad_flag):
        label = False
        reason = '短文：包含违禁词'
        line = '\t'.join(parts)
        sku2result[sku] = (line, label, reason)
        continue
      line = '\t'.join(parts)
      sku2result[sku] = (line, label, reason)
      good += 1

  print('too long or short %s' % cnt)
  print('good sku: %d' % good)
  print('==========%d========' % len(list(skus)))

  with open(final_file + ".valid.withPic", 'w', encoding='utf-8') as sw:
    id = 0
    for sku, res in sku2result.items():
      id += 1
      sps = sku2sps[sku]
      if sku in validsku2pics:
        line, label, reason = res
        # sps = sku2sps[sku]
        pics = validsku2pics[sku]
        # parts = line.split('\t')
        # title = parts[8]
        # if (len(title.replace(' ', '')) < 6 or len(title.replace(' ', '')) > 10):
        #   cnt += 1
        #   print(title)
        # if (len(title.split(' ')) != 3):
        #   cnt += 1
        #   print(title)
        sw.write(str(label) + '\t' + reason + '\t' + str(sps) + '\t' +  str(id) + line + '\t' + '\t'.join(pics) + '\n')
      else:
        reason = '未获取sku对应Pic信息'
        sw.write(str(label) + '\t' + reason + '\t' + str(sps) + '\t' +  str(id) + line + '\t' + '\n')
        print(sku)

def titleLength(title):
  pat = r'[\u4e00-\u9fa5]'
  tmp = re.sub(pattern=pat, repl='aa', string=title.replace(' ', ''))
  return str(len(tmp) / 2)

def isCountry(token):
  if token in ['美国', '意大利']:
    return True
  return False

if __name__ == "__main__":
  # test_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\dev.sku'
  # test_modelPrediction_file = r'D:\Research\Projects\AlphaWriting\model\data\dev.titlePrediction-v1.0'
  # test_modelPrediction_new_file = r'D:\Research\Projects\AlphaWriting\script\data\dev.titlePrediction.new'

  if (False):
    test_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\test.sku'
    test_titlePrediction_file = r'D:\Research\Projects\AlphaWriting\model\data\test.titlePrediction-v1.0'
    test_writingPrediction_file = r'D:\Research\Projects\AlphaWriting\model\data\test.prediction.t2s_forceDecoding-v1.0'
    test_titlePrediction_new_file = r'D:\Research\Projects\AlphaWriting\model\data\test.titlePrediction-v2.0'

    data_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\jiayongdianqi.writing.reduced.joinKB'

    sku2info = loadAllData(data_file)
    sku2prediction = loadSku2prediction(test_file, test_titlePrediction_file, test_writingPrediction_file)
    rule_based_title_generation(test_file, test_titlePrediction_new_file, sku2info, sku2prediction)

  if (False):  # xuanpin
    test_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin_goodsku1000\xuanpin.merge.sku'
    test_titlePrediction_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin_goodsku1000\xuanpin.merge.sku.prediction.title.v1.0'
    # test_writingPrediction_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin_goodsku1000\xuanpin.merge.sku.prediction.writing'
    # test_writingPrediction_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin_goodsku1000\xuanpin.merge.sku.prediction.writing.v0.1.4'
    test_writingPrediction_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin_goodsku1000\xuanpin.merge.sku.prediction.writing.v0.1.5-beam4-2sps'
    # test_titlePrediction_new_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin_goodsku1000\xuanpin.merge.sku.prediction.v2.0'
    # test_titlePrediction_new_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin_goodsku1000\xuanpin.merge.sku.prediction.v0.1.4'
    test_titlePrediction_new_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin_goodsku1000\xuanpin.merge.sku.prediction.v0.1.5-beam4-2sps'

    sku_basic_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.skuInfo'
    sku_ext_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.ext.0408.filtered'
    sku_spec_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.spec.0408.filtered.concat'

    black_list_file = r'D:\Research\Projects\AlphaWriting\data\blackList.txt'
    bad_sku_list_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin_goodsku1000\xuanpin.merge.sku.bad'

    black_list = loadSkus(black_list_file)
    bad_sku_list = loadSkus(bad_sku_list_file)
    sku_list = loadSkus(test_file)

    sku2info = {}
    sku2info = add_sku_basic_info(sku_list, sku_basic_info_file, sku2info)
    sku2info = add_kb(sku_list, sku_ext_info_file, sku2info)  # jiayongdianqi.kb.ext.0408.filtered
    sku2info = add_kb(sku_list, sku_spec_info_file, sku2info)  # jiayongdianqi.kb.spec.0408.filtered.concat
    sku2prediction = loadSku2prediction(sku_list, test_titlePrediction_file, test_writingPrediction_file)
    rule_based_title_generation_forNewSku(sku_list, test_titlePrediction_new_file, sku2info, sku2prediction, black_list,
                                          bad_sku_list)

  if (False):  # xuanpin2
    test_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.sku'
    test_titlePrediction_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.prediction.title.KB0.1.2'
    test_writingPrediction_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.prediction.writing.KB0.1.5'
    test_titlePrediction_new_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.prediction.KB0.1.5'

    sku_basic_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.skuInfo'
    sku_ext_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.ext.0408.filtered'
    sku_spec_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.spec.0408.filtered.concat'

    black_list_file = r'D:\Research\Projects\AlphaWriting\data\blackList.txt'
    bad_sku_list_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.sku.bad'

    black_list = loadSkus(black_list_file)
    bad_sku_list = loadSkus(bad_sku_list_file)
    sku_list = loadSkus(test_file)

    sku2info = {}
    sku2info = add_sku_basic_info(sku_list, sku_basic_info_file, sku2info)
    sku2info = add_kb(sku_list, sku_ext_info_file, sku2info)  # jiayongdianqi.kb.ext.0408.filtered
    sku2info = add_kb(sku_list, sku_spec_info_file, sku2info)  # jiayongdianqi.kb.spec.0408.filtered.concat
    sku2prediction = loadSku2prediction(sku_list, test_titlePrediction_file, test_writingPrediction_file)
    rule_based_title_generation_forNewSku(sku_list, test_titlePrediction_new_file, sku2info, sku2prediction, black_list,
                                          bad_sku_list)

  if (False):  # xuanpin3
    test_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.validSku'
    test_titlePrediction_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.prediction.title.KB0.1.2'
    test_writingPrediction_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.prediction.writing.KB0.1.5'
    test_titlePrediction_new_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.prediction.KB0.1.5'

    sku_basic_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.skuInfo'
    sku_ext_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.ext.0408.filtered'
    sku_spec_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.spec.0408.filtered.concat'

    black_list_file = r'D:\Research\Projects\AlphaWriting\data\blackList.txt'
    bad_sku_list_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.sku.bad'

    black_list = loadSkus(black_list_file)
    bad_sku_list = loadSkus(bad_sku_list_file)
    sku_list = loadSkus(test_file)

    if (True):
      sku2info = {}
      sku2info = add_sku_basic_info(sku_list, sku_basic_info_file, sku2info)
      sku2info = add_kb(sku_list, sku_ext_info_file, sku2info)  # jiayongdianqi.kb.ext.0408.filtered
      sku2info = add_kb(sku_list, sku_spec_info_file, sku2info)  # jiayongdianqi.kb.spec.0408.filtered.concat
      sku2prediction = loadSku2prediction(sku_list, test_titlePrediction_file, test_writingPrediction_file)
      rule_based_title_generation_forNewSku(sku_list, test_titlePrediction_new_file, sku2info, sku2prediction,
                                            black_list, bad_sku_list)

    # valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.havePicture'
    # valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\toProvide'
    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.havePicture.label.valid.forCheck'
    # valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.havePicture.label.valid'
    with open(valid_sku_file, 'r', encoding='utf-8') as f:
      # valid_skus = [w for w in list(set([_.strip().split('\t')[0].strip() for _ in f])) if w not in bad_sku_list]
      # valid_skus = list(set([_.strip().split('\t')[1].strip() for _ in f]))
      validsku2pics = {}
      while (True):
        line = f.readline()
        if (not line):
          break
        line = line.strip('\n')
        parts = line.split('\t')
        sku = parts[1].strip()
        pic = parts[3].strip()
        if (sku not in validsku2pics):
          validsku2pics[sku] = []
        validsku2pics[sku].append(pic)
      print('===========%d==========' % len(validsku2pics.items()))
    filtData(test_titlePrediction_new_file, validsku2pics, bad_sku_list)

  if (False):  # xuanpin4
    spNum = 3
    root = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4'
    test_file = os.path.join(root, r'xuanpin4.validSku')
    test_titlePrediction_file = os.path.join(root, r'xuanpin4.prediction.title.KB0.1.2')
    test_writingPrediction_file = os.path.join(root, r'sp'+str(spNum), 'xuanpin4.prediction.writing.KB0.1.5')
    # test_writingPrediction_file = os.path.join(root, r'sp'+str(spNum), 'xuanpin4.prediction.writing.punc.KB0.1.5')
    test_titlePrediction_new_file = os.path.join(root, r'sp'+str(spNum), 'xuanpin4.prediction.KB0.1.5')
    sku_basic_info_file = os.path.join(root, r'xuanpin4.basicInfo')
    sku_kb_info_file = os.path.join(root, r'xuanpin4.table')
    black_list_file = r'D:\Research\Projects\AlphaWriting\data\blackList.txt'
    bad_sku_list_file = os.path.join(root, r'sp'+str(spNum), 'xuanpin4.sku.bad')

    black_list = loadSkus(black_list_file)
    bad_sku_list = loadSkus(bad_sku_list_file)
    sku_list = loadSkus(test_file)

    if (True):
      sku2info = load_kb_for_dataset(sku_basic_info_file, sku_kb_info_file)
      sku2prediction = loadSku2prediction(sku_list, test_titlePrediction_file, test_writingPrediction_file)
      rule_based_title_generation_forNewSku(sku_list, test_titlePrediction_new_file, sku2info, sku2prediction,
                                            black_list, bad_sku_list)

    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\xuanpin4.havePicture'
    with open(valid_sku_file, 'r', encoding='utf-8') as f:
      validsku2pics = {}
      while (True):
        line = f.readline()
        if (not line):
          break
        line = line.strip('\n')
        parts = line.split('\t')
        sku = parts[0].strip()
        pic = parts[1].strip()
        if (sku not in validsku2pics):
          validsku2pics[sku] = []
        validsku2pics[sku].append(pic)
      print('===========%d==========' % len(validsku2pics.items()))
    spFile = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\xuanpin4.constraint-5sellingPoints-ocr-largeIdf'
    sku2sps = loadSku2SellPoints(sku_list, spFile)
    filtData(test_titlePrediction_new_file, validsku2pics, sku2sps, bad_sku_list)

  if (False):  # xuanpin5
    spNum = 3
    root = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3'
    test_file = os.path.join(root, r'xuanpin3.validSku')
    test_titlePrediction_file = os.path.join(root, r'xuanpin3.prediction.title.KB0.1.2')
    test_writingPrediction_file = os.path.join(root, r'sp'+str(spNum), 'xuanpin3.prediction.writing.punc.KB0.1.5')
    test_titlePrediction_new_file = os.path.join(root, r'sp'+str(spNum), 'xuanpin3.prediction.KB0.1.5')
    sku_basic_info_file = os.path.join(root, r'xuanpin3.basicInfo')
    sku_kb_info_file = os.path.join(root, r'xuanpin3.table')
    black_list_file = r'D:\Research\Projects\AlphaWriting\data\blackList.txt'
    bad_sku_list_file = os.path.join(root, r'sp'+str(spNum), 'xuanpin3.sku.bad')

    black_list = loadSkus(black_list_file)
    bad_sku_list = loadSkus(bad_sku_list_file)
    sku_list = loadSkus(test_file)

    if (True):
      sku2info = load_kb_for_dataset(sku_basic_info_file, sku_kb_info_file)
      sku2prediction = loadSku2prediction(sku_list, test_titlePrediction_file, test_writingPrediction_file)
      rule_based_title_generation_forNewSku(sku_list, test_titlePrediction_new_file, sku2info, sku2prediction,
                                            black_list, bad_sku_list)

    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin5\xuanpin5.sku'
    with open(valid_sku_file, 'r', encoding='utf-8') as f:
      validsku2pics = {}
      while (True):
        line = f.readline()
        if (not line):
          break
        sku = line.strip('\n')

        if (sku not in validsku2pics):
          validsku2pics[sku] = []
        validsku2pics[sku].append('')
      print('===========%d==========' % len(validsku2pics.items()))
    filtData(test_titlePrediction_new_file, validsku2pics, bad_sku_list)

  if (False):  # xuanpin6
    spNum = 3
    root = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin6'
    test_file = os.path.join(root, r'xuanpin6.validSku')
    test_titlePrediction_file = os.path.join(root, r'xuanpin6.prediction.title.KB0.1.2')
    # test_writingPrediction_file = os.path.join(root, r'sp'+str(spNum), 'xuanpin6.prediction.writing.KB0.1.5')
    test_writingPrediction_file = os.path.join(root, r'sp'+str(spNum), 'xuanpin6.prediction.writing.punc.KB0.1.5')
    test_titlePrediction_new_file = os.path.join(root, r'sp'+str(spNum), 'xuanpin6.prediction.KB0.1.5')
    sku_basic_info_file = os.path.join(root, r'xuanpin6.basicInfo')
    sku_kb_info_file = os.path.join(root, r'xuanpin6.table')
    black_list_file = r'D:\Research\Projects\AlphaWriting\data\blackList.txt'
    bad_sku_list_file = os.path.join(root, r'sp'+str(spNum), 'xuanpin6.sku.bad')

    black_list = loadSkus(black_list_file)
    bad_sku_list = loadSkus(bad_sku_list_file)
    sku_list = loadSkus(test_file)

    if (True):
      sku2info = load_kb_for_dataset(sku_basic_info_file, sku_kb_info_file)
      sku2prediction = loadSku2prediction(sku_list, test_titlePrediction_file, test_writingPrediction_file)
      rule_based_title_generation_forNewSku(sku_list, test_titlePrediction_new_file, sku2info, sku2prediction,
                                            black_list, bad_sku_list)

    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin6\xuanpin6.havePicture.new'
    with open(valid_sku_file, 'r', encoding='utf-8') as f:
      validsku2pics = {}
      while (True):
        line = f.readline()
        if (not line):
          break
        line = line.strip('\n')
        parts = line.split('\t')
        sku = parts[0].strip()
        pic = parts[1].strip()
        if (sku not in validsku2pics):
          validsku2pics[sku] = []
        validsku2pics[sku].append(pic)
      print('===========%d==========' % len(validsku2pics.items()))
    spFile = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin6\xuanpin6.constraint-5sellingPoints-ocr-largeIdf'
    sku2sps = loadSku2SellPoints(sku_list, spFile)
    filtData(test_titlePrediction_new_file, validsku2pics, sku2sps, bad_sku_list)

  if (True):  # chuju
    # spNum = 3
    # root = r'/export/homes/baojunwei/alpha-w-pipeline'
    # domain = 'chuju'
    # dataset = 'test'
    # model_version = 'KB0.1.6'
    root = sys.argv[1]
    domain = sys.argv[2]  # 'chuju'
    dataset = sys.argv[3]  # 'test'
    model_version = sys.argv[4]

    test_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.validSku')
    test_titlePrediction_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.prediction.title')
    test_writingPrediction_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.prediction.writing')
    test_titlePrediction_new_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.prediction')
    sku_basic_info_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.basicInfo')
    sku_kb_info_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.table')
    black_list_file = os.path.join(root, 'data', 'blackList.txt')
    bad_sku_list_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.sku.bad')

    black_list = loadSkus(black_list_file)
    bad_sku_list = loadSkus(bad_sku_list_file)
    sku_list = loadSkus(test_file)

    if (True):
      sku2info = load_kb_for_dataset(sku_basic_info_file, sku_kb_info_file)
      sku2prediction = loadSku2prediction(sku_list, test_titlePrediction_file, test_writingPrediction_file)
      brandName2aliases = {'松下': ['松下', 'Panasonic'],
                           'Panasonic': ['松下', 'Panasonic'],
                           '欧思嘉': ['欧思嘉', 'oucica'],
                           'oucica': ['欧思嘉', 'oucica'],
                           '阿里斯顿': ['阿里斯顿', 'ariston'],
                           'ariston': ['阿里斯顿', 'ariston'],
                           '现代': ['现代', 'HYUNDAI'],
                           'HYUNDAI': ['现代', 'HYUNDAI'],
                           '法仕康': ['法仕康', '法士康'],
                           '法士康': ['法仕康', '法士康'],
                           '卡杰诗': ['卡杰诗', 'kgc'],
                           'kgc': ['卡杰诗', 'kgc']
                           }
      rule_based_title_generation_forNewSku(sku_list, test_titlePrediction_new_file, sku2info, sku2prediction,
                                            black_list, bad_sku_list, model_version, brandName2aliases)

    all_sku_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.validSku')
    with open(all_sku_file, 'r', encoding='utf-8') as f:
      validsku2pics = {}
      while (True):
        line = f.readline()
        if (not line):
          break
        line = line.strip('\n')
        parts = line.split('\t')
        sku = parts[0].strip()
        pic = ''
        if (sku not in validsku2pics):
          validsku2pics[sku] = []
        validsku2pics[sku].append(pic)
      print('===========%d==========' % len(validsku2pics.items()))

    spFile = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.table')
    spFile = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.constraint-5sellingPoints-ocr-largeIdf')
    sku2sps = loadSku2SellPoints(sku_list, spFile)
    filtData_allDoamin(test_titlePrediction_new_file, validsku2pics, sku2sps, bad_sku_list, model_version)

  if (False):  # chuju_joinPic
    # spNum = 3
    #root = r'/export/homes/baojunwei/alpha-w-pipeline'
    # root = sys.argv[1]
    # domain = sys.argv[2] #'chuju'
    # dataset = sys.argv[3] #'test'
    root = sys.argv[1]
    domain = sys.argv[2]  # 'chuju'
    dataset = sys.argv[3]  # 'test'
    model_version = sys.argv[4]
    test_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.validSku')
    test_titlePrediction_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.prediction.title')
    test_writingPrediction_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.prediction.writing')
    test_titlePrediction_new_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.prediction')
    sku_basic_info_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.basicInfo')
    sku_kb_info_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.table')
    black_list_file = os.path.join(root, 'data', 'blackList.txt')
    bad_sku_list_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.sku.bad')

    black_list = loadSkus(black_list_file)
    bad_sku_list = loadSkus(bad_sku_list_file)
    sku_list = loadSkus(test_file)

    if (False):
      sku2info = load_kb_for_dataset(sku_basic_info_file, sku_kb_info_file)
      sku2prediction = loadSku2prediction(sku_list, test_titlePrediction_file, test_writingPrediction_file)
      rule_based_title_generation_forNewSku(sku_list, test_titlePrediction_new_file, sku2info, sku2prediction,
                                            black_list, bad_sku_list)

    all_sku_file = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.pic')
    with open(all_sku_file, 'r', encoding='utf-8') as f:
      validsku2pics = {}
      while (True):
        line = f.readline()
        if (not line):
          break
        line = line.strip('\n')
        parts = line.split('\t')
        sku = parts[0].strip()
        pic = parts[1].strip()
        if (sku not in validsku2pics):
          validsku2pics[sku] = []
        validsku2pics[sku].append(pic)
      print('===========%d==========' % len(validsku2pics.items()))

    spFile = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.table')
    spFile = os.path.join(root, 'data', domain + '_' + dataset, dataset + '.constraint-5sellingPoints-ocr-largeIdf')
    sku2sps = loadSku2SellPoints(sku_list, spFile)
    filtData_allDoamin_joinPic(test_titlePrediction_new_file, validsku2pics, sku2sps, bad_sku_list, model_version)

  if (False):  # alphaSales
    spNum = 3
    root = r'D:\Research\Projects\AlphaWriting\model\data\alphaSales'
    test_file = os.path.join(root, r'alphaSales.validSku')
    test_titlePrediction_file = os.path.join(root, r'alphaSales.prediction.title.KB0.1.2')
    # test_writingPrediction_file = os.path.join(root, r'sp'+str(spNum), 'alphaSales.prediction.writing.KB0.1.5')
    test_writingPrediction_file = os.path.join(root, r'sp'+str(spNum), 'alphaSales.prediction.writing.punc.KB0.1.5')
    test_titlePrediction_new_file = os.path.join(root, r'sp'+str(spNum), 'alphaSales.prediction.KB0.1.5')
    sku_basic_info_file = os.path.join(root, r'alphaSales.basicInfo')
    sku_kb_info_file = os.path.join(root, r'alphaSales.table')
    black_list_file = r'D:\Research\Projects\AlphaWriting\data\blackList.txt'
    bad_sku_list_file = os.path.join(root, r'sp'+str(spNum), 'alphaSales.sku.bad')

    black_list = loadSkus(black_list_file)
    bad_sku_list = loadSkus(bad_sku_list_file)
    sku_list = loadSkus(test_file)

    if (True):
      sku2info = load_kb_for_dataset(sku_basic_info_file, sku_kb_info_file)
      sku2prediction = loadSku2prediction(sku_list, test_titlePrediction_file, test_writingPrediction_file)
      rule_based_title_generation_forNewSku(sku_list, test_titlePrediction_new_file, sku2info, sku2prediction,
                                            black_list, bad_sku_list)

    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\alphaSales\alphaSales.validSku'
    with open(valid_sku_file, 'r', encoding='utf-8') as f:
      validsku2pics = {}
      while (True):
        line = f.readline()
        if (not line):
          break
        line = line.strip('\n')
        sku = line
        pic = ''
        if (sku not in validsku2pics):
          validsku2pics[sku] = []
        validsku2pics[sku].append(pic)
      print('===========%d==========' % len(validsku2pics.items()))
    spFile = r'D:\Research\Projects\AlphaWriting\model\data\alphaSales\alphaSales.constraint-5sellingPoints-ocr-largeIdf'
    sku2sps = loadSku2SellPoints(sku_list, spFile)
    filtData(test_titlePrediction_new_file, validsku2pics, sku2sps, bad_sku_list)

  if (False):  # daren
    test_file = r'D:\Research\Projects\AlphaWriting\model\data\daren\sku.txt'
    sku_basic_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.skuInfo'
    sku_list = loadSkus(test_file)
    sku2info = {}
    sku2info = add_sku_basic_info(sku_list, sku_basic_info_file, sku2info)
    get_input_info_for_new_sku(test_file, sku_list, sku2info)

  if (False):  # filter picture
    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.havePicture.label'
    sku2pics = {}
    with open(valid_sku_file, 'r', encoding='utf-8') as f:
      # valid_skus = [w for w in list(set([_.strip().split('\t')[0] for _ in f])) if w not in bas_sku_list]
      while (True):
        line = f.readline()
        if (not line):
          break
        line = line.strip()
        parts = line.split('\t')
        id = parts[0].strip()
        sku = parts[1].strip()
        pic = parts[4].strip()

        label = parts[7].strip() if len(parts) == 8 else ''
        if sku not in sku2pics:
          sku2pics[sku] = []
        if (label == '1' or label == '2'):
          sku2pics[sku].append((pic, label))
    tmp = {}
    for key, values in sku2pics.items():
      tmp[key] = sorted(values, key=lambda x: x[1])
    sku2pics = tmp

    result = [(sku, values) for sku, values in sku2pics.items() if
              len(values) >= 3 and any([l == '1' for l in [t[1] for t in values]])]
    print(len(result))
    with open(valid_sku_file + '.valid', 'w', encoding='utf-8') as sw:
      for res in result:
        sku, picsLabel = res
        sw.write(sku + "\t" + ' ||| '.join([_[0] for _ in picsLabel]) + '\t' + ' ||| '.join(
          [_[1] for _ in picsLabel]) + '\t' + str(len(picsLabel)) + '\n')

    with open(valid_sku_file + '.valid.forCheck', 'w', encoding='utf-8') as sw:
      id = 1
      for res in result:
        sku, picsLabel = res
        for pic, lab in picsLabel:
          sw.write(
            str(id) + '\t' + sku + '\t' + r'https://item.jd.com/' + sku + '.html' + '\t' + pic + '\t' + lab + '\n')
        id += 1

  if (False):
    with open(r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\toProvide', 'r', encoding='utf-8') as sr:
      with open(r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\toProvide.withIndex', 'w',
                encoding='utf-8') as sw:
        index = 0
        lastSku = ''
        while (True):
          line = sr.readline().replace(' .html', '.html')
          if (not line):
            break
          sku = line.split('\t')[0]
          if (lastSku != sku):
            index += 1
            lastSku = sku
          sw.write(str(index) + '\t' + line)

  if (False):  # xunapin2 join have picture
    bas_sku_list = loadSkus(r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.sku.bad')

    # valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.havePicture'
    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.havePicture.label.valid'

    with open(valid_sku_file, 'r', encoding='utf-8') as f:
      valid_skus = [w for w in list(set([_.strip().split('\t')[0].strip() for _ in f])) if w not in bas_sku_list]
      # valid_skus = list(set([_.strip().split('\t')[0] for _ in f]))

    final_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.prediction.KB0.1.5'
    filter(final_file, valid_skus)

  if (False):  # xunapin3 join have picture
    bas_sku_list = loadSkus(r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.sku.bad')

    # valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.havePicture'
    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.havePicture.label.valid'

    with open(valid_sku_file, 'r', encoding='utf-8') as f:
      valid_skus = [w for w in list(set([_.strip().split('\t')[0].strip() for _ in f])) if w not in bas_sku_list]
      # valid_skus = list(set([_.strip().split('\t')[0] for _ in f]))

    final_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3.prediction.KB0.1.5'
    filter(final_file, valid_skus)



