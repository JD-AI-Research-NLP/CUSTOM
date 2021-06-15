import json
import sys
import requests
import os
import time
#r1 = open('/export/scratch/wangyifan/kb_write/shouji/data/shouji/k.txt','r')
#r2 = open('/export/scratch/wangyifan/kb_write/shouji/data/shouji_xuanpin4/xuanpin4','r')
#r3 = open('/export/scratch/wangyifan/kb_write/shouji/data/shouji/cat_id', 'r')
#w1 = open('./testtest','w')
root = sys.argv[1]
domain = sys.argv[2]
dataset = sys.argv[3]
folder = sys.argv[4]
input_file_1 = os.path.join(root, 'data', domain + '_' + folder, 'k.txt')
input_file_2 = os.path.join(root, 'data', domain+'_'+dataset, dataset)
input_file_3 = os.path.join(root, 'data', domain + '_' + folder, 'cat_id')
out_1 = os.path.join(root, 'data', domain+'_'+dataset, dataset+'.validSku')
out_2 = os.path.join(root, 'data', domain+'_'+dataset, dataset+'.table')
out_3 = os.path.join(root, 'data', domain + '_' + folder, domain + '.kb.skuId2popId')
out_5 = os.path.join(root, 'data', domain+'_'+dataset, dataset+'.fakeWriting')
out_6 = os.path.join(root, 'data', domain+'_'+dataset, dataset+'.constraint-5sellingPoints-ocr-largeIdf')

w1  = open(out_1,'w')
w2 = open(out_2, 'w')
w3 = open(out_3, 'w')
w5 = open(out_5, 'w')
w6 = open(out_6, 'w')
k_dic = {}
dic_kv={}
dic_sort = {}
for line in open(input_file_1).readlines():
  line1 = line.strip()
  k_dic[line1] = 1
dic_cat = {}
tmp_id=''
for line in open(input_file_3).readlines():
  line1 = line.strip().split('\t')
  if len(line1)!=2:
    continue
  dic_cat[line1[0]] = line1[1]
count = 0
for sku in open(input_file_2).readlines():
    time.sleep(0.1)
    count += 1
    if count % 200 == 0:
        print(count)
    dic_sort={}
    dic_kv={}
    sku = sku.strip().split('\t')[0]
    try:
      r = requests.get('http://alpha-sales-proxy-prod.jd.com/getKb?sku=' + sku + '&env=jimi1')#or '&env=env'
      res = (r.json())
    #exit(0)
      for k in res:
        if type(res[k]) == dict:
           for j in res[k]:
              if k+'_'+j in k_dic:
                 dic_kv[k+'_'+j] = res[k][j]
        else:
           if k in k_dic:
              dic_kv[k] = res[k]
        if k=='item_id':
            tmp_id = res[k]
      dic_sort['SkuName'] = dic_kv['sku_name']
      dic_sort['BrandName_full'] = dic_kv['brandname_full']
      dic_sort['BrandName_cn'] = dic_kv['brandname_cn']
      dic_sort['BrandName_en'] = dic_kv['brandname_en']
      dic_sort['CategoryL1'] = dic_cat[dic_kv['cate1_cd']]
      dic_sort['CategoryL2'] = dic_cat[dic_kv['cate2_cd']]
      dic_sort['CategoryL3'] = dic_cat[dic_kv['cate3_cd']]
      dic_kv.pop('sku_name')
      dic_kv.pop('brandname_full')
      dic_kv.pop('brandname_cn')
      dic_kv.pop('brandname_en')
      dic_kv.pop('cate1_cd')
      dic_kv.pop('cate2_cd')
      dic_kv.pop('cate3_cd')
      for k in dic_kv:
        if k not in dic_sort:
            dic_sort[k] = dic_kv[k]
      list_t = []
      for k,v in dic_sort.items():
          list_t.append(k+" $$ "+v)
      w1.write(str(sku)+'\n')
      w2.write(' ## '.join(list_t)+'\n')
      w3.write(str(sku)+'\t'+str(tmp_id)+'\n')
      w5.write(str(1)+'\n')
      w6.write('\n')
      w1.flush()
      w2.flush()
      w5.flush()
      w6.flush()
      w3.flush()
    except:
     continue
