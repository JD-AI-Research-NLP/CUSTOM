import requests
import json
import os
import sys

def post_writing(sku, title, writing, version):
  HEADERS = {'Content-Type': 'application/json'}
  url = 'http://alpha-writing-platform.jd.com/insert_material_info'
  data = {'sku': sku, 'title': title, 'document': writing, 'modelVersion': version}
  result = requests.post(url=url, headers=HEADERS, data=json.dumps(data)).text
  res = json.loads(result)
  return res

def get_writing(categoryIds, modelVersion):
  #url = 'http://alpha-writing-platform.jd.com/get_writable_sku_list'
  url = 'http://alpha-writing-platform.jd.com/get_writable_sku_list'
  data = {"categoryIds": categoryIds, "modelVersion": modelVersion}
  result = requests.get(url=url, params=data).text
  res = json.loads(result)
  #res = result.json()
  return res



if __name__ == '__main__':
  root = sys.argv[1]
  domain = sys.argv[2]
  dataset = sys.argv[3]
  max_num = int(sys.argv[4])
  modelVersion = sys.argv[5]
  catIds_path = os.path.join(root, 'data', domain+"_rawData", "thirdids_supported")
  r1 = open(catIds_path,'r')
  cats=','.join([i.strip().split('\t')[0] for i in r1.readlines()])

  file = os.path.join(root, 'data', domain + '_' + dataset, dataset)

  #print(post_or_get)
  print(file)
  results = []
  results = get_writing(cats, modelVersion)
  with open(file, 'w', encoding='utf-8') as w:
    num = 0
    for res in results['message']['result']:
      num += 1
      w.write(res['sku'] + '\t' + res['categoryId'] + '\t' + res['category'] + '\n')
      if num >= max_num:
        break
