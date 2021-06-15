from collections import Counter
import re
import sys
import os
import jieba
import math
dic1={}
lines=[]
t=0
word_dict={}
dic_key_word={}
def handle_kv_to(file_path_ckg0,path_blacklist,path_output):
  list_file = os.listdir(file_path_ckg0)
  for file_ in list_file:
      if file_.find('ckg0')==0:
         handle_kv(file_path_ckg0+'/'+file_,path_blacklist,path_output)
def handle_kv(file1,path_blacklist,path_output):
  r1 = open(file1,'r')
  r2 = open(path_blacklist,'r')
  black={}
  for line in r2.readlines():
     line1 = line.strip()
     black[line1]=1
#r1 = open('jsq','r')
  cat = file1.split('.')[1].strip()
  print(cat)
 #w = open('jsq_item.txt','w')
  w = open(path_output+'/'+cat+'.item.txt','w')
  t=[]
  sumnum=0
  tt=[]
  ss=[]
  key1=''
  for line in r1.readlines():
     t.append(line)
  for line in t:
     if line[0]!='\t' and line!='\n':
        key1  = line.split('\t')[0]
        sumnum = int(line.split('\t')[2])
        tt=[]
        ss=[]
     elif line!='\n':
        key  = line.split('\t')[1].split(':')[0].strip()
        if key not in black:
          num = int(line.split('\t')[2].strip())
          tt.append(key)
          ss.append(float(num)/sumnum)
     else:
        if key1.find('SkuName')>=0 or key1.find('Labels')>=0:
          sumnum=0
          key1=''
          tt=[]
          ss=[]
          continue
        if len(tt)==0:
          sumnum=0
          key1=''
          tt=[]
          ss=[]
          continue
        w.write(key1+'\n')
        for index in range(len(tt)):
          w.write(tt[index]+'\t'+str(ss[index])+'\t')
        w.write('\n')
        sumnum=0
        key1=''
        tt=[]
        ss=[]
def kuozhan(key,sent,dic_key_word):
  global t
  t+=1
  jie_list = jieba.cut(sent)
  if key not in dic_key_word:
    dic_key_word[key]=[]
  for word in jie_list:
    if len(word)>1:
      dic_key_word[key].append(word)
      if word not in word_dict:
        word_dict[word]=0
      word_dict[word]+=1
def tf_idf(file_path,file1,output_path):
  r1 = open(file_path+'/'+file1,'r')
  print(file1)
  cat = file1.split('.')[0].strip()
  print(cat)
  try:
     r2 = open(file_path+'/'+cat+'.data','r')
  except:
     return
#r2 = open('jsq_train_8k5','r')
  #w1 = open('kv_train_item_del','w')
  w = open(output_path+'/'+cat+'_nle','w')
  dic1={}
  lines=[]
  t=0
  word_dict={}
  dic_key_word={}
  for line in r1.readlines():
   line1 = line.strip()
   lines.append(line1)
  for i in range(len(lines)):
    if (i+1)%2==1:
      dic1[lines[i]]=[]
      values = lines[i+1].split('\t')
      for v in range(len(values)):
        if (v+1)%2==1:
          dic1[lines[i]].append(values[v])
  print(dic1)
  total=0
  right=0
  word_k=''
  isOneKey=[]
  for line in r2.readlines():
    line = line.strip()
    sents = re.split('。|！|？',line)
    for sent in sents:
      word_k=''
      total+=1
      Flag = False
      #w1.write(sent+'\t')
      for key in dic1:
        for word in dic1[key]:
          if sent.find(word)>=0:
            if word_k=='':
              word_k=key
            if word_k!=key:
              isOneKey.append('-1')
      if '-1' in isOneKey:
        #w1.write('\n')
        isOneKey=[]
        word_k=''
        continue
      for key in dic1:
        for word in dic1[key]:
          if sent.find(word)>=0:
            kuozhan(key,sent,dic_key_word)
            #w1.write(key+'\t'+word)
      #w1.write('\n')
  print(len(dic_key_word))
  for key in dic_key_word:
    w.write(key+'\n')
    w.write('\t'.join(dic1[key])+'\n')
    t1 = len(dic_key_word[key])
    c = Counter(dic_key_word[key]).most_common()
    dic_t={}
    for (k,v) in c:

    #w.write(k+':'+str(float(v)/t1)+'\t'+str(math.log(float(t)/word_dict[k],1.1))+'\t')
    #dic_t[k]=(float(v)/t1)*(math.log(float(t)/word_dict[k],2))
      t2=0
      for key1 in dic_key_word:
         if k in dic_key_word[key1]:
           t2+=1
           dic_t[k]=(float(v)/t1)*(math.log(float(len(dic_key_word))/t2,2))
    c1 = sorted(dic_t.items(), key = lambda x: x[1], reverse=True)
    for (k,v) in c1:
      w.write(k+':'+str(v)+'\t')
    w.write('\n')
    w.write('\n')
def tf_idf_to(file_path,output_path):
   list_file = os.listdir(file_path)
   for file1 in list_file:
       if file1.find('item.txt')>0:
           tf_idf(file_path,file1,output_path)
def get_cat(output_path,file1,file2,file3):
  r1 = open(file1,'r')
  r2 = open(file2,'r')
  r3 = open(file3,'r')
  title=[]
  text=[]
  table=[]
  for line in r3.readlines():
    line = line.strip()
    table.append(line)
  for line in r1.readlines():
    line = line.strip()
    title.append(line)
  for line in r2.readlines():
    line = line.strip()
    text.append(line)
  assert(len(title)==len(text)==len(table))
  dic_cat_text={}
  for i in range(len(text)):
    cat = table[i].split('CategoryL3 $$')[1].split('##')[0].strip()
    if cat not in dic_cat_text:
       dic_cat_text[cat]=[]
    dic_cat_text[cat].append(text[i])
  for dic in dic_cat_text:
    texts = dic_cat_text[dic]
    if '/' in dic:
      dic = dic.replace('/','_')
    w = open(output_path+'/'+dic+'.data','w')
    for text in texts:
      w.write(text.strip()+'\n')
def get_idf(table_file,text_file,idf_out):
   r1 = open(table_file,'r')
   r2 = open(text_file,'r')
   dic_cat={}
   lines=[]
   for line in r1.readlines():
     j = line.strip()
     lines.append(j)
     cat = j.split('CategoryL3 $$')[1].split('##')[0].strip()
     if '/' in cat:
       cat = cat.replace('/','_')
     if cat not in dic_cat:
       dic_cat[cat]=[]
   titles=[]
   for line in r2.readlines():
     j = line.strip()
     titles.append(j)
   num=0
   print(dic_cat)
   for n in range(len(titles)):
     cat = lines[n].split('CategoryL3 $$')[1].split('##')[0].strip()
     if '/' in cat:
       cat = cat.replace('/','_')
     dic_cat[cat].append(titles[n])
   num=0
   for cat in dic_cat:
     to=0
     word_dict={}
     word_dict_tf={}
     word_dict_idf={}
     for sent in dic_cat[cat]:
       to+=1
       line_list = set(jieba.lcut(sent))
       for l in line_list:
         if l not in word_dict:
           word_dict[l]=1
         else:
           word_dict[l]+=1
     for key in word_dict:
       if word_dict[key]>0:
         word_dict_idf[key] =math.log(to/word_dict[key],2)
         word_dict_tf[key] = word_dict[key]
     d = sorted(word_dict_idf.items(),key = lambda x: x[1],reverse=True)
     w = open(idf_out+'/'+cat+'_idf','w')
     num+=1
     for (k,v) in d:
       w.write(k+'\t'+str(v)+'\t'+str(word_dict_tf[k])+'\n')
if __name__=='__main__':
    #dataset = 'chuju'
    #root = '/export/homes/baojunwei/alpha-w-pipeline/data/ckg'
    #root2 = '/export/homes/baojunwei/alpha-w-pipeline/data/'
    root = os.path.join(sys.argv[1], 'data')
    print(root)
    ckgRoot = os.path.join(root, 'ckg')
    domain = sys.argv[2]
    try:
     os.mkdir(os.path.join(ckgRoot,'tmp'))
    except:
     pass
    if not os.path.isdir(os.path.join(ckgRoot, domain, 'ckg1')):
      os.mkdir(os.path.join(ckgRoot, domain, 'ckg1'))

    print(os.path.join(root, domain,'train.table'))
    get_idf(os.path.join(root, domain,'train.table'),os.path.join(root, domain, 'train.text'),os.path.join(ckgRoot, domain,'ckg1'))
    handle_kv_to(os.path.join(ckgRoot, domain, 'ckg0'), os.path.join(ckgRoot, 'blacklist'), os.path.join(ckgRoot,'tmp'))
    get_cat(os.path.join(ckgRoot,'tmp'), os.path.join(root, domain,'train.title'), os.path.join(root, domain, 'train.text'), os.path.join(root, domain,'train.table'))
    tf_idf_to(os.path.join(ckgRoot,'tmp'), os.path.join(ckgRoot, domain,'ckg1')) 
