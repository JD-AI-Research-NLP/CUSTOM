import requests
import json
import sys
import os
path = sys.argv[1]
domain = sys.argv[2]
xuanpin = sys.argv[3]
root_path = os.path.join(path, 'data', domain+'_'+xuanpin)
data_path = os.path.join(root_path, 'res.filterBySupportedCat3')
r1 = open(data_path,'r')
def post_writing(sku, title, writing, version):
  HEADERS = {'Content-Type': 'application/json'}
  # url = 'http://alpha-writing.jd.local/insert_material_info'
  url = 'http://alpha-writing-platform.jd.com/insert_material_info'
  data = {'sku': sku, 'title': title, 'document': writing, 'modelVersion': version}
  result = requests.post(url=url, headers=HEADERS, data=json.dumps(data)).text
  res = json.loads(result)
  return res
for line in r1.readlines():
  line1 = line.strip().split('\t')
  print(line1[0])
  if len(line1[-1])<55:
    continue
  if len(line1)!=3:
    continue
  post_writing(line1[0], line1[1], line1[2], 'KB2.0')
