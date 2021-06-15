import sys
import os
r1 = open(sys.argv[1],'r')
r2 = open(sys.argv[2],'r')
w1 = open(sys.argv[3],'w')

# 获取valid table的路径
out_data_path = sys.argv[3]
valid_kb_path = os.path.join('/'.join(out_data_path.split('/')[:-2]), 'valid_kb')

# 读取kb
valid_kb = list()
with open(valid_kb_path, 'r') as f:
  for i in f:
    valid_kb.append(i.strip())

dic1={}
for line in r2.readlines():
  line1  =line.strip().split('\t')
  if line1[0] not in dic1:
     dic1[line1[0]]=[]
  if line1[-1] not in dic1[line1[0]]:
    dic1[line1[0]].append(line1[-1])
for line in r1.readlines():
  line1  =line.strip().split('\t')
  if line1[0] not in dic1:
    kv = line1[1]
    to_write = kv.split('##')
    final_kb = list()
    w1.write(to_write[0] + ' ##' )
    for i in range(1,len(to_write)):
      buf_key = to_write[i].split('$$')[0].strip()
      if buf_key in valid_kb:
        final_kb.append(to_write[i])
    w1.write('##'.join(final_kb))
    w1.write('\n')
  else:
    kv = line1[1]
    to_write = kv.split('##')
    name = kv.split('SkuName $$')[1].strip().split('##')[0].strip()
    if len(dic1[line1[0]])>10:
      name =name + '。'.join([a for a in dic1[line1[0]][:50] if len(a)>0])
    else:
      name = name + '。'.join(dic1[line1[0]])
    #for a in name:
    #  w1.write(line1[0]+'\t'+a+'\n')
    w1.write('SkuName $$ '+name+' ##')
    final_kb = list()
    for i in range(1,len(to_write)):
      buf_key = to_write[i].split('$$')[0].strip()
      if buf_key in valid_kb:
        final_kb.append(to_write[i])
    w1.write('##'.join(final_kb))
    w1.write('\n')