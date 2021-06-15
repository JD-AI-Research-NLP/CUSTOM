import os
import re
import glob
import sys

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


def keepKV(key, value):
  if(key != "" and value != "" and value != "--" and key != "Labels"):
    return True
  return False

def load_kbInfo(file):
  id2kb = {}
  with open(file, 'r', encoding='utf-8') as f:
    id = ""
    writingID = ""
    sku = ""
    title = ""
    text = ""
    kb = {}
    while 1:
      line = f.readline()
      if not line:
        print(id)
        break

      line = line.strip('\n')
      if line.startswith('AlphaWriting'):
        id = line.split('-')[1]
      if line.startswith('WritingID'):
        writingID = int(line[11:])
      if line.startswith('Sku'):
        sku = int(line[5:])
      if line.startswith('Title:'):
        title = line[7:]
      if line.startswith('Text:'):
        text = line[6:]
      if line.startswith('KB:'):
        kv = line[4:].split(' ||| ')
        key = kv[0].strip()
        value = kv[1].strip()
        if (len(kv) == 2 and keepKV(key, value)):
          values = kb.get(key, None)
          if not values:
            kb[key] = []
          kb[key].append(value)
      if line == "":
        if text != "" and len(kb) != 0:
          id2kb[id]=(kb, writingID, sku, text, title)
        else:
          print(id)
        kb = {}
        text = ""
  return id2kb

def load_sku_basic_info(sku_basic_info_file):
  sku2basicInfo = {}
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
      sku2basicInfo[Sku]={'Sku': Sku,
                          'SkuName': SkuName,
                          'Brand_id': Brand_id,
                          'BrandName_en': BrandName_en,
                          'BrandName_cn': BrandName_cn,
                          'BrandName_full': BrandName_full,
                          'CategoryL1_id': CategoryL1_id,
                          'CategoryL1': CategoryL1,
                          'CategoryL2_id': CategoryL2_id,
                          'CategoryL2': CategoryL2,
                          'CategoryL3_id': CategoryL3_id,
                          'CategoryL3': CategoryL3}
    return sku2basicInfo


def groupByCategoryL3(id2kb, distinct_level, file):
  '''distinct level: (id, writingID, sku)'''
  cat2attr = {} #cat3: (attr_distribution, item_cnt)
  groupKeys = set()


  for id, tuple in id2kb.items():
    kb, writingID, sku, text, title = tuple

    groupKey = ""
    if distinct_level == 'id':
      groupKey = id
    elif distinct_level == 'writingID':
      groupKey = writingID
    elif distinct_level == 'sku':
      groupKey = sku
    if groupKey in groupKeys:
      continue

    groupKeys.add(groupKey)
    cat3 = kb['CategoryL3'][0]
    #cat3 = kb['BrandName_full'][0]
    # cat3 = "all"

    if cat3 not in cat2attr:
      attr_dis = {}
      cat2attr[cat3] = [attr_dis, 0]
    for key, value in kb.items():
      if not cat2attr[cat3][0].get(key, None):
        cat2attr[cat3][0][key] = [0, 0, {}] #distinct attribute cnt; value_cnt weighted attribute cnt
      cat2attr[cat3][0][key][0] = cat2attr[cat3][0][key][0] + 1
      cat2attr[cat3][0][key][1] = cat2attr[cat3][0][key][1] + len(value)
      for v in value:
        if not cat2attr[cat3][0][key][2].get(v, None):
          cat2attr[cat3][0][key][2][v] = 0
        cat2attr[cat3][0][key][2][v] = cat2attr[cat3][0][key][2][v] + 1

    cat2attr[cat3][1] = cat2attr[cat3][1] + 1

  for cat, tuple in cat2attr.items():
    cat = cat.replace(r'/', '_')
    attr_dict, cnt = tuple
    sort_attr_dict1 = sorted(attr_dict.items(), key = lambda x: x[1][0], reverse=True)
    sort_attr_dict2 = sorted(attr_dict.items(), key = lambda x: x[1][1], reverse=True)
    with open(file + cat + ".distinct_" + distinct_level + ".from_" + str(cnt) + "_instance", 'w', encoding='utf-8') as w:
      for attr, cnt_attr_values in sort_attr_dict1:
        w.write(attr + "\t:\t" + str(cnt_attr_values[0]) + "\n")
        v_dist = sorted(cnt_attr_values[2].items(), key=lambda x:x[1], reverse=True)
        for v, c_cnt in v_dist:
          w.write("\t" + v + ":\t" + str(c_cnt) + "\n")
        w.write("\n")

    # with open(file + cat + ".distinct_" + distinct_level + ".from_" + str(cnt) + "_instance" + ".value_cnt_weighted", 'w', encoding='utf-8') as w:
    #   for attr, cnt_attr_values in sort_attr_dict2:
    #     w.write(attr + "\t:\t" + str(cnt_attr_values[1]) + "\n")
    #     v_dist = sorted(cnt_attr_values[2].items(), key=lambda x:x[1], reverse=True)
    #     for v, c_cnt in v_dist:
    #       w.write("\t" + v + ":\t" + str(c_cnt) + "\n")
    #     w.write("\n")

def groupByCategoryL3_forLabel(id2kb, distinct_level, file):
  '''distinct level: (id, writingID, sku)'''
  cat2attr = {} #cat3: (attr_distribution, item_cnt)
  groupKeys = set()


  for id, tuple in id2kb.items():
    kb, writingID, sku, text, title = tuple

    groupKey = ""
    if distinct_level == 'id':
      groupKey = id
    elif distinct_level == 'writingID':
      groupKey = writingID
    elif distinct_level == 'sku':
      groupKey = sku
    if groupKey in groupKeys:
      continue

    groupKeys.add(groupKey)
    cat3 = kb['CategoryL3'][0]
    if(cat3 not in ['冰箱','净水器','空调','洗衣机','油烟机']):
      continue
    #cat3 = kb['BrandName_full'][0]
    # cat3 = "all"


    if not cat2attr.get(cat3, None):
      attr_dis = {}
      cat2attr[cat3] = [attr_dis, 0]
    for key, value in kb.items():
      if not cat2attr[cat3][0].get(key, None):
        cat2attr[cat3][0][key] = [0, 0, {}] #distinct attribute cnt; value_cnt weighted attribute cnt
      cat2attr[cat3][0][key][0] = cat2attr[cat3][0][key][0] + 1
      cat2attr[cat3][0][key][1] = cat2attr[cat3][0][key][1] + len(value)
      for v in value:
        if not cat2attr[cat3][0][key][2].get(v, None):
          cat2attr[cat3][0][key][2][v] = 0
        cat2attr[cat3][0][key][2][v] = cat2attr[cat3][0][key][2][v] + 1

    cat2attr[cat3][1] = cat2attr[cat3][1] + 1

  for cat, tuple in cat2attr.items():
    cat = cat.replace(r'/', '_')
    attr_dict, cnt = tuple
    sort_attr_dict1 = sorted(attr_dict.items(), key = lambda x: x[1][0], reverse=True)
    sort_attr_dict2 = sorted(attr_dict.items(), key = lambda x: x[1][1], reverse=True)
    with open(file + cat + ".distinct_" + distinct_level + ".from_" + str(cnt) + "_instance", 'w', encoding='utf-8') as w:
      for attr, cnt_attr_values in sort_attr_dict1:
        v_dist = sorted(cnt_attr_values[2].items(), key=lambda x:x[1], reverse=True)
        w.write(attr + "\t" + v_dist[0][0] + "\n")


def data_filter(ocr_file_ori, ocr_file, sku2info, cat3s):
  with open(ocr_file_ori, 'r', encoding='utf-8') as f:
    with open(ocr_file, 'w', encoding='utf-8') as w:
      while(True):
        line = f.readline()
        if(not line):
          break
        line = line.strip('\n')
        parts = line.split('\t')
        sku = parts[0]
        ocr = parts[3]
        if(sku not in sku2info):
          continue
        info = sku2info[sku]
        cat = info['CategoryL3']
        #if (cat not in cat3s):
        #  continue
        w.write(line + '\n')
def load_cat2funcValues(file_prefix, cat, roundNum, maxNum):
  # file_name = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.cat2funcWords.' + cat
  file_name = file_prefix + '.' +cat + '.round' + str(roundNum)

  d = dict()
  cnt = 0
  for k, v in [_.strip().split('\t') for _ in open(file_name, 'r', encoding='utf-8')]:
    if len(k) > 15:
      continue
    v = int(v)
    if(cnt >= maxNum):
      break
    cnt += 1
    d[k] = v
  return d

def load_cat2funcPattern(file_prefix, cat, roundNum, maxNum):
  # file_name = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.cat2funcPatterns.' + cat
  file_name = file_prefix + '.' +cat + '.round' + str(roundNum)

  d = dict()
  cnt = 0
  for k, v in [_.strip().split('\t') for _ in open(file_name, 'r', encoding='utf-8')]:
    if len(k) > 15:
      continue
    v = int(v)
    if(cnt >= maxNum):
      break
    cnt += 1
    d[k] = v
  return d

def dump_functional_values_round_0_jiayongdianqi(ocr_file, output_file, sku2info, ckg):
  import re
  from collections import Counter
  pats = [r'(搭载|采用|具有|内置|外置|拥有|支持)(\S+)(技术|科技|引擎|功能|设计)',
          r'(搭载|采用|具有|内置|外置|拥有|支持)(\S+)($)',
          r'(^)(\S+)(技术|科技|引擎|功能|设计)',
          #r'([\u4e00-\u9fa5]*)([A-z0-9/!@#$%^&*()_+°\-]+)([\u4e00-\u9fa5]*)',
          # r'([\u4e00-\u9fa5]*)([A-z0-9/()+°\.\-%#@&℃]+)([\u4e00-\u9fa5]*)'
          ]

  cat2vs = {}
  cat2valueOcrs = {}
  index = 0

  with open(ocr_file, 'r', encoding='utf-8') as f:
    while(True):
      line = f.readline()
      if(not line):
        break
      line = line.strip()
      index += 1
      parts = line.split('\t')
      sku = parts[0]
      ocr = parts[3]
      if(sku not in sku2info):
        continue
      info = sku2info[sku]
      cat = info['CategoryL3']
      if(cat not in ['冰箱', '洗衣机', '空调', '净水器', '油烟机']):
        continue

      ocr_subs = re.split(r'[,，。；;!！？?、:()*%#@|\[\]]', ocr)  # ocr.split(',，。；;!！？?')
      for ocr_sub in ocr_subs:
        for pat in pats:
          m = re.search(pat, ocr_sub)
          if m == None:
            continue
          v = m.groups()[1]
          if(len(v) < 2 or len(v) > 10):
            continue
          m1 = re.match(r'.+的(.+)$', v)
          if m1 != None:
            v = m1.groups()[0]
          v= v.strip('的')
          if cat not in cat2vs:
            cat2vs[cat] = Counter()
            cat2valueOcrs[cat] = {}
          cat2vs[cat].update([v])
          if v not in cat2valueOcrs[cat]:
            cat2valueOcrs[cat][v] = []
          cat2valueOcrs[cat][v].append(ocr_sub)


          if(index % 10000 == 0):
            print(index)
          break

  for cat, vs in cat2vs.items():
    with open(output_file + '.' + cat + '.round0' , 'w', encoding='utf-8') as w:
      if cat == '油烟机':
        teSeTuiJian = [x for y in ckg.get_cat_ckg(cat)['特色推荐'] for x in y]
      if cat == '洗衣机':
        teSeTuiJian = [x for y in ckg.get_cat_ckg(cat)['特色推荐'] for x in y]
      if cat == '冰箱':
        teSeTuiJian = [x for y in ckg.get_cat_ckg(cat)['特色推荐'] for x in y]
      if cat == '净水器':
        tmp = ckg.get_cat_ckg(cat)
        teSeTuiJian = []
        if('特色推荐' in tmp):
          teSeTuiJian = [x for y in tmp['特色推荐'] for x in y]
        if('产品特点' in tmp):
          teSeTuiJian.extend([x for y in tmp['产品特点'] for x in y])
        teSeTuiJian = list(set(teSeTuiJian))
      if cat == '空调':
        tmp = ckg.get_cat_ckg(cat)
        teSeTuiJian = []
        if ('特色推荐' in tmp):
          teSeTuiJian = [x for y in tmp['特色推荐'] for x in y]
        if ('产品特点' in tmp):
          teSeTuiJian.extend([x for y in tmp['产品特点'] for x in y])
        teSeTuiJian = list(set(teSeTuiJian))
      for v in teSeTuiJian:
        w.write(v + '\t' + str(10000) +'\n')

      for v, cnt in vs.most_common():
        w.write(v + '\t' + str(cnt) +'\n')
        # for sub_ocr in cat2valueOcrs[cat][v]:
        #   w.write('ocr:{}\n'.format(sub_ocr))

def dump_functional_values_round_0(ocr_file, output_file, sku2info, ckg, pats):
  import re
  from collections import Counter
  cat2vs = {}
  cat2valueOcrs = {}
  index = 0
  with open(ocr_file, 'r', encoding='utf-8') as f:
    while(True):
      line = f.readline()
      if(not line):
        break
      line = line.strip('\n')
      index += 1
      parts = line.split('\t')
      sku = parts[0]
      ocr = parts[3]
      if(sku not in sku2info):
        continue
      info = sku2info[sku]
      cat = info['CategoryL3']
      # if(cat not in ['冰箱', '洗衣机', '空调', '净水器', '油烟机']):
      #   continue

      ocr_subs = re.split(r'[,，。；;!！？?、:()*%#@|\[\]]', ocr)  # ocr.split(',，。；;!！？?')
      for ocr_sub in ocr_subs:
        for pat in pats:
          m = re.search(pat, ocr_sub)
          if m == None:
            continue
          v = m.groups()[1]
          if(len(v) < 2 or len(v) > 10):
            continue
          m1 = re.match(r'.+的(.+)$', v)
          if m1 != None:
            v = m1.groups()[0]
          v= v.strip('的')
          if cat not in cat2vs:
            cat2vs[cat] = Counter()
            cat2valueOcrs[cat] = {}
          cat2vs[cat].update([v])
          if v not in cat2valueOcrs[cat]:
            cat2valueOcrs[cat][v] = []
          cat2valueOcrs[cat][v].append(ocr_sub)


          if(index % 10000 == 0):
            print(index)
          break

  for cat, vs in cat2vs.items():
    cat = cat.replace('/', '_')
    with open(output_file + '.' + cat + '.round0' , 'w', encoding='utf-8') as w:
      kb = ckg.get_cat_ckg(cat)
      teSeTuiJian = []
      if('特色推荐' in kb):
        teSeTuiJian.extend([x for y in kb['特色推荐'] for x in y])
      if('产品特点' in kb):
        teSeTuiJian.extend([x for y in kb['产品特点'] for x in y])
      if ('产品特色' in kb):
        teSeTuiJian.extend([x for y in kb['产品特色'] for x in y])
      tmp = [k.split('_')[1] for k, v in kb.items() if k.startswith('特色功能') and v == '支持']
      if (len([tmp]) > 0):
        teSeTuiJian.extend(tmp)
      teSeTuiJian = list(set(teSeTuiJian))    
      #tmp = ckg.get_cat_ckg(cat)
      #teSeTuiJian = []
      #if('特色推荐' in tmp):
      #  teSeTuiJian = [x for y in tmp['特色推荐'] for x in y]
      #if('产品特点' in tmp):
      #  teSeTuiJian.extend([x for y in tmp['产品特点'] for x in y])
      #teSeTuiJian = list(set(teSeTuiJian))

      for v in teSeTuiJian:
        w.write(v + '\t' + str(10000) +'\n')

      for v, cnt in vs.most_common():
        w.write(v + '\t' + str(cnt) +'\n')
        # for sub_ocr in cat2valueOcrs[cat][v]:
        #   w.write('ocr:{}\n'.format(sub_ocr))


def dump_functional_pattern_round_N(ocr_file, input_file, output_file, sku2info, roundNum):
  import re
  from collections import Counter
  cat2functionalValues = {}
  cat2vs = {}
  for cat in ['冰箱', '洗衣机', '空调', '净水器', '油烟机']:
    cat2functionalValues[cat] = load_cat2funcValues(input_file, cat, roundNum-1, maxNum=100)
    index = 0

  pats = []#r'^|,(.+),|$'
  with open(ocr_file, 'r', encoding='utf-8') as f:
    while (True):
      line = f.readline()
      if (not line):
        break
      line = line.strip()
      index += 1
      parts = line.split('\t')
      sku = parts[0]
      ocr = parts[3]
      if (sku not in sku2info):
        continue
      info = sku2info[sku]
      cat = info['CategoryL3']
      if (cat not in ['冰箱', '洗衣机', '空调', '净水器', '油烟机']):
        continue
      for pat in cat2functionalValues[cat]:
        ocr_subs = re.split(r'[,，。；;!！？?、:()*%#@|\[\]]', ocr) #ocr.split(',，。；;!！？?')
        for ocr_sub in ocr_subs:
          if ocr_sub == '':
            continue
          try:
            m = re.search(pat, ocr_sub)
          except:
            continue

          if m == None:
            continue
          v = re.sub(pat, '(\S+)', ocr_sub)
          if v == '(\S+)':
            continue
          v = v.strip('的')
          if cat not in cat2vs:
            cat2vs[cat] = Counter()
          cat2vs[cat].update([v])

          if (index % 10000 == 0):
            print(index)


  for cat, vs in cat2vs.items():
    with open(output_file + '.' + cat + '.round' + str(roundNum), 'w', encoding='utf-8') as w:
      for v, cnt in vs.most_common():
        w.write(v + '\t' + str(cnt) +'\n')

def dump_functional_values_round_N(ocr_file, output_file, input_file, sku2info, roundNum):
  import re
  from collections import Counter
  cat2Pats = {}
  cat2vs = {}

  for cat in ['冰箱', '洗衣机', '空调', '净水器', '油烟机']:
    cat2Pats[cat] = load_cat2funcPattern(input_file, cat, roundNum, 100)
    index = 0

  cat2vs = {}
  index = 0

  with open(ocr_file, 'r', encoding='utf-8') as f:
    while(True):
      line = f.readline()
      if(not line):
        break
      line = line.strip()
      index += 1
      parts = line.split('\t')
      sku = parts[0]
      ocr = parts[3]
      if(sku not in sku2info):
        continue
      info = sku2info[sku]
      cat = info['CategoryL3']
      if(cat not in ['冰箱', '洗衣机', '空调', '净水器', '油烟机']):
        continue
      ocr_subs = re.split(r'[,，。；;!！？?、:()*%#@|\[\]]', ocr)  # ocr.split(',，。；;!！？?')
      for ocr_sub in ocr_subs:
        for pat in cat2Pats[cat]:
          try:
            m = re.search(pat, ocr_sub)
          except:
            continue

          if m == None:
            continue
          v = m.groups()[0]

          m1 = re.match(r'.+的(.+)$', v)
          if m1 != None:
            v = m1.groups()[0]
          v= v.strip('的')
          if (len(v) < 2):
            continue
          if cat not in cat2vs:
            cat2vs[cat] = Counter()
          cat2vs[cat].update([v])

          if(index % 10000 == 0):
            print(index)

  for cat, vs in cat2vs.items():
    with open(output_file + '.' + cat + '.round' + str(roundNum), 'w', encoding='utf-8') as w:
      for v, cnt in vs.most_common():
        w.write(v + '\t' + str(cnt) +'\n')


if __name__ == "__main__":
  #root = r'/export/homes/baojunwei/alpha-w-pipeline/data'
  #domain = 'chuju'
  root = os.path.join(sys.argv[1], 'data')
  domain = sys.argv[2]
  dataset = sys.argv[3]
  folder = sys.argv[4]
  cat3s = sys.argv[5:]

  if (True):
    os.makedirs(os.path.join(root, 'funcWord', domain))
    sku_basic_info_file = os.path.join(root, domain + '_' + folder, domain +'.kb.skuInfo')
    sku2info = load_sku_basic_info(sku_basic_info_file)
    #ocr_file_ori = os.path.join(root, domain, dataset + '.ocr')
    ocr_file_ori = os.path.join(root, domain, dataset+'.ocr')
    #ocr_file = os.path.join(root, domain, dataset + '.ocr.filtered')
    ocr_file = os.path.join(root, domain, dataset + '.ocr.filtered')
    function_value_file = os.path.join(root, 'funcWord', domain, domain + '.cat2funcWords')
    function_value_pattern_file = os.path.join(root, 'funcWord', domain, domain + '.cat2funcPatterns')
    ckg_path = os.path.join(root, 'ckg', domain, 'ckg0/ckg0.*')
    ckg = CKG(ckg_path)

    #cat3s = ['茶壶', '保温杯', '玻璃杯', '茶杯']
    # cat3s = ['冰箱', '洗衣机', '空调', '净水器', '油烟机']
    data_filter(ocr_file_ori, ocr_file, sku2info, cat3s)
    roundNum = 0
    pats = [r'(搭载|采用|具有|内置|外置|拥有|支持)(\S+)(技术|科技|引擎|功能|设计)',
            r'(搭载|采用|具有|内置|外置|拥有|支持)(\S+)($)',
            r'(^)(\S+)(技术|科技|引擎|功能|设计)',
            # r'([\u4e00-\u9fa5]*)([A-z0-9/!@#$%^&*()_+°\-]+)([\u4e00-\u9fa5]*)',
            # r'([\u4e00-\u9fa5]*)([A-z0-9/()+°\.\-%#@&℃]+)([\u4e00-\u9fa5]*)'
            ]
    print('Round-{}============================'.format(roundNum))
    print('Functional Value Mining: Round-{}============================'.format(roundNum))
    dump_functional_values_round_0(ocr_file, function_value_file, sku2info, ckg, pats)
    for roundNum in range(1, 0):
      print('Round-{}============================'.format(roundNum))
      print('Functional Pattern Mining: Round-{}============================'.format(roundNum))
      dump_functional_pattern_round_N(ocr_file, function_value_file, function_value_pattern_file, sku2info, roundNum)
      print('Functional Value Mining: Round-{}============================'.format(roundNum))
      dump_functional_values_round_N(ocr_file, function_value_file, function_value_pattern_file, sku2info, roundNum)
