import sys
import pdb
import os

root = sys.argv[1]
domain = sys.argv[2]
folder = sys.argv[3]

infile = os.path.join(root, 'data', domain, 'train.table')
cat_id_file = os.path.join(root, 'data', domain + '_' + folder, 'train.gdm')
supported_cat3s = os.path.join(root, 'data', domain + '_' + folder, 'thirdids_supported')

cat3name2id = {}
with open(cat_id_file, 'r', encoding='utf-8') as f:
  for line in f:
    parts = line.strip().split('\t')
    if not parts[7] in cat3name2id:
      cat3name2id[parts[7]] = parts[6]


dic = {}
with open(infile, 'r', encoding='utf-8') as f:
  for line in f:
    parts = line.split(' ## ')
    cat3 = parts[6]
    kv = cat3.split(' $$ ')
    #pdb.set_trace()
    if len(kv) != 2 or kv[0] != 'CategoryL3':
      continue
    if kv[1] not in dic:
      dic[kv[1]] = 0
    dic[kv[1]] += 1

sortedDic = sorted(dic.items(), key=lambda x:x[1], reverse=True)
with open(supported_cat3s, 'w') as w:
  for kv in sortedDic:
    if(kv[1] > 2000):
      w.write(cat3name2id[kv[0]] + '\t' + kv[0] + '\n')

print('Total:%d'%(sum([v for k, v in sortedDic])))
