#encoding=utf-8
import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
import requests
import json
import re
import os
import copy
import jieba
def get_sku_itemid(file1, file2, file3):
  r1 = open(file1, 'r')
  r2 = open(file2, 'r')
  w1 = open(file3, 'w')
  sku_itemid = {}
  for line in r2.readlines():
    line1 = line.strip().split('\t')
    if line1[0] not in sku_itemid:
      sku_itemid[line1[0]] = line1[1]
  for line in r1.readlines():
    line1 = line.strip()
    if line1 in sku_itemid:
      w1.write(line1 + '\t' + sku_itemid[line1] + '\n')
    else:
      w1.write(line1 + '\t' + line1 + '\n')
  w1.flush()
  
def get_img_from_html(text):
  tmp = re.findall("jfs/t.*?\.", text)
  imgs = []
  for i in tmp:
    tmp_url = 'http://img30.360buyimg.com/sku/' + i + 'jpg'
    imgs.append(tmp_url)
  return imgs
        
def get_img(file1, file2):
  r1 = open(file1, 'r')
  w1 = open(file2, 'w')
  #print(file2)
  num = 0
  dic = {}
  for line in r1.readlines():
    if num % 100 == 0:
      print(num)
    num += 1
    # time.sleep(.5)
    line1 = line.strip().split('\t')
    p_id = line1[1]
    sku = line1[0]
    #url = "https://cd.jd.com/description/channel?skuId=" + line1[1] + "&mainSkuId=" + line1[1]
    url = "http://alpha-writing-platform.jd.com/product/queryBigField?sku="+sku
#     url = "http://alpha-writing.jd.local/product/material/image/query?sku="+sku
    try:
      res = requests.get(url).text
      tmp = re.findall("jfs/t.*?\.", res)
      for i in tmp:
         tmp_url = 'http://img30.360buyimg.com/sku/' + i + 'jpg'
         w1.write(line1[0] + '\t' + line1[1] + '\t' + tmp_url + '\n')
         dic[line1[0]] = 1
    except:
      pass
  print(len(dic))
  w1.flush()

# discard
def get_img_back(file1, file2):
  r1 = open(file1, 'r')
  w1 = open(file2, 'w')
  #print(file2)
  num = 0
  dic = {}
  for line in r1.readlines():
    if num % 100 == 0:
      print(num)
    num += 1
    # time.sleep(.5)
    line1 = line.strip().split('\t')
    p_id = line1[1]
    sku = line1[0]
    #url = "https://cd.jd.com/description/channel?skuId=" + line1[1] + "&mainSkuId=" + line1[1]
    url = "http://alpha-writing-platform.jd.com/product/queryBigField?sku="+sku
    res = requests.get(url).text
    try:
      if 'background-image:url' in res:
        res_l = res.split('background-image:url(')
        for i in range(1, len(res_l)):
          tmp_url = ""
          tmp = res_l[i]
          for w in tmp:
            if w != ')':
              tmp_url += w
            else:
              break
          if not tmp_url.startswith("http"):
            tmp_url = "https:" + tmp_url
          if len(tmp_url) < 15:
            continue
          w1.write(line1[0] + '\t' + line1[1] + '\t' + tmp_url + '\n')
          dic[line1[0]] = 1
      elif 'src' in res:
        res_l =res.split('src=\\"')
        for i in range(1,len(res_l)):
          tmp_url = ""
          tmp = res_l[i]
          for w in tmp:
            if w != '\\':
              tmp_url += w
            else:
              break
          if not tmp_url.startswith("http"):
            tmp_url = "https:" + tmp_url
          if len(tmp_url) < 15:
            continue
          w1.write(line1[0] + '\t' + line1[1] + '\t' + tmp_url + '\n')
          dic[line1[0]] = 1
      elif 'data-lazyload=\\"' in res:
        res_l = res.split('data-lazyload=\\"')
        for i in range(1, len(res_l)):
          tmp_url = ""
          tmp = res_l[i]
          for w in tmp:
            if w != '\\':
              tmp_url += w
            else:
              break
          if not tmp_url.startswith("http"):
            tmp_url = "https:" + tmp_url
          if len(tmp_url) < 15:
            continue
          w1.write(line1[0] + '\t' + line1[1] + '\t' + tmp_url + '\n')
          dic[line1[0]] = 1
    except:
      print(tmp_url)
      pass
  print(len(dic))
  w1.flush()

def get_ocr(file1, file2):
  skus = []
  r1 = open(file1, 'r')
  w1 = open(file2, 'w')
  num = 0
  for line in r1.readlines():
    if num % 100 == 0:
      print(num)
    num += 1
    try:
      line1 = line.strip().split('\t')
      url = line1[2]
      sku = line1[0]
      p_id = line1[1]
      TOKEN = "0c7bada0-cede-45a3-84d2-76052d3d3182"
      img_url = url.strip()
      url = "http://g.jsf.jd.local/com.jd.cv.jsf.ComputerVisionService/OCR-GPU-V5/recognizeCharacter/40431/jsf/29000"
      data = [TOKEN, img_url, {"imgusecache": "true", "isusecache": "true"}] 
      res = json.loads(requests.post(url, json.dumps(data)).text)
      ocr = []
      dic = {}
      if 'texts' in res:
        ocr = res['texts']
        dic[sku] = ocr
        d = json.dump(dic, w1)
        w1.write('\n')
    except:
      print(sku)
      pass
  w1.flush()

def clean_text(text):
  # keep English, digital and Chinese
  comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5^*-+/]') # 保留（*-+.）特殊字符，例如6gb+128gb
  return comp.sub('', text)

# 长度和非中文过滤
def filter_text(data):
  zh_pattern = re.compile(r'[\u4e00-\u9fa5]+') # 判断至少有一个中文字符
  # mix_pattern = re.compile(r'[0-9\u4e00-\u9fa5]+') # 判断是否纯字母或只有字母和特殊字符
  clean = clean_text(data)
  if len(clean) < 3 or len(clean) > 30 or clean.isdigit() or not zh_pattern.search(clean):
    return ''
  return clean

def get_word_scores(path):
  all_word_scores = dict()
  with open(path, 'r') as f:
    for i in f:
      split = i.strip().split('\t')
      all_word_scores[split[0]] = split[1]
  return all_word_scores

def cal_word_score(all_word_scores, text):
  text = text.lower()
  text_split = jieba.lcut(text)
  ocr_tfidf = 0.0
  for w in text_split:
    ocr_tfidf += float(all_word_scores.get(w, 0.0))
  ocr_tfidf = ocr_tfidf / len(text_split)
  return ocr_tfidf

def ocr_json_2_txt(file1, file2, file3, file4):
  all_word_scores = get_word_scores(file4)
  r1 = open(file1, 'r')
  ad_word = []
  r2 = open(file2, 'r')
  for line in r2.readlines():
    line1 = line.strip()
    ad_word.append(line1)
  w4 = open(file3, 'w')
  dict_width = {}
  dict_height = {}
  dict_pro = {}
  for line in r1.readlines():
    sku = list(json.loads(line.strip()).keys())[0]
    line = list(json.loads(line.strip()).values())[0]
    flag_ocr_have_other_goods = False
    #print(line)
    buf_data = list()  # 用来暂存解析出的ocr
    try:
      for temp in line:
        try:
          area = temp['area']
          width = area['width']
          height = area['height']
          pro = temp['probability']
          text = temp['text']
          # 首先判断是否有广告词
          for i in ad_word:
            if i in text:
              flag_ocr_have_other_goods = True
              break
          if flag_ocr_have_other_goods == True:
            break
          if (type(pro)) != type(1.1) or float(pro) < 0 or float(pro) > 1.0 or float(pro) < 0.5:
            continue
          text = filter_text(text)
          score = cal_word_score(all_word_scores, text)
          if not text == '' and score > 0:
            buf = str(sku) + '\t' + str(float(width) * float(height)) + '\t' + str(width) + '\t' + str(height) + '\t' + str(
                pro) + '\t' + str(score) + '\t' + str(text)
            buf_data.append(buf)
        except Exception as e1:
          continue
    except Exception as e:
      print(e)
      print(line)
    if not flag_ocr_have_other_goods:
      for item in buf_data:
        w4.write(item + '\n')
        

def handle_line(file1, file2):
  r1 = open(file1, 'r')
  w1 = open(file2, 'w')
  for line in r1.readlines():
    line1 = line.strip().split('\t')
    if line1[0].find('||') > 0:
      lines = line1[0].split('||')
      for l in lines:
        w1.write(l + '\t' + '\t'.join(line1[1:]) + '\n')
    else:
      w1.write(line)

def func1(string):
  if '.' in string:
     index_dot = string.index('.')
     if (index_dot-1)>=0 and '0'<=string[index_dot-1]<='9' and (index_dot+1)<len(string) and '0'<=string[index_dot+1]<='9':
         return string
     else: 
         return string.replace('.','')
  return string  

def handle_ocr_step1(file1, file2):
  sku = ''
  area = []
  pro = []
  text = []
  skus = []
  num1 = 0
  num2 = 0
  w = open(file2, 'w')
  r = open(file1, 'r')
  sku=''
  for line in r.readlines():
    line1 = line.strip().split('\t')
    if len(line1) != 7:
      continue
    list_l = re.split(',|，|；|;|、|。|:|：|!|！|"|“|”|/', line1[-1])
    for l in list_l:
              #print(l)
       l = func1(l)
       l = filter_text(l)
       if l == '':
         continue
       if '+' in l:
          ll = l.split('+')
          for l1 in ll:
             w.write(line1[0] + '\t' + str(line1[1]) + '\t' + str(line1[4]) + '\t' + str(line1[5]) + '\t' + str(l1) + '\n')
       if len(l)>0:
          w.write(line1[0] + '\t' + str(line1[1]) + '\t' + str(line1[4]) + '\t' + str(line1[5]) + '\t' + str(l) + '\n')

if __name__ == '__main__':
  root=os.path.join(sys.argv[1], 'data')
  domain = sys.argv[2]
  #root = r'/data0/baojunwei/alpha-w-pipeline/data'
  dataset = sys.argv[3]
  tmp = os.path.join(root, 'tmp_train' if dataset == 'train' else 'tmp')
  if not os.path.isdir(tmp):
    os.mkdir(tmp)
  folder = sys.argv[4]
  in_folder = domain + "_" + dataset if dataset != 'train' else domain
  in_file_name = '.sku' if dataset == 'train' else '.validSku'

  
  get_sku_itemid(os.path.join(root, in_folder, "total" + in_file_name) if dataset=='train' else os.path.join(root, in_folder, dataset + in_file_name),
                 os.path.join(root, domain + "_" + folder, domain + '.kb.skuId2popId'), os.path.join(tmp, 'sku_itemid'))
  sku_itemid_url_file = os.path.join(tmp, 'sku_itemid_url')
  print('fuck one')
  get_img(os.path.join(tmp, 'sku_itemid'), sku_itemid_url_file)
  print('fuck two')
  sku_itemid_ocr_file = os.path.join(tmp, 'sku_itemid_ocr')
  #exit(0)
 
  os.system('bash ./ocr.sh' + ' ' + tmp + ' ' + sku_itemid_url_file + ' ' + sku_itemid_ocr_file)
  ocr_json_2_txt(sku_itemid_ocr_file, os.path.join(root, 'ad_keywords'),
                 os.path.join(tmp, 'txt_ocr'), os.path.join(root, 'tf_idf_nonorm'))
  #exit(0)
  handle_line(os.path.join(tmp, 'txt_ocr'), os.path.join(tmp, 'txt_ocr_1'))
  handle_ocr_step1(os.path.join(tmp, 'txt_ocr_1'), os.path.join(root, in_folder, dataset + '.ocr'))
  os.system('cp ' + os.path.join(tmp, 'sku_itemid_url') +' ' + os.path.join(root, in_folder, dataset + '.img'))
