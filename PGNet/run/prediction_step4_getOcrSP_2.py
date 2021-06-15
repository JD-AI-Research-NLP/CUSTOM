import jieba
import glob
import sys
import os
r111 = open(sys.argv[1],'r')
r11 = open(sys.argv[2],'r')
w2 = open(sys.argv[4],'w')
w1 = open(sys.argv[3],'w')
root=sys.argv[7]
r8  = open(sys.argv[6],'r')
r9 = open(sys.argv[8],'r')
r10 = open(sys.argv[10],'r')
dic={}
all_title=[]
name_title={}
cat_title={}
to_dic=0

for line in r10.readlines():
   line1 = line.strip()
   cat = line1.split('CategoryL3 $$')[1].split('##')[0].strip()
   name = line1.split('BrandName_cn $$')[1].split('##')[0].strip()
   cat_title[cat]=1
   name_title[name]=1
for line in r9.readlines():
   all_title.append(line.strip())
   line1 = line.strip().split(' ')
   if len(line1)==3:
      dic[line1[1]]=1
def find_sale(input_file,output_file):
  r1 = open(input_file,'r')
  w1 = open(output_file,'w')
  dic1={}
  sku={}
  for line in r1.readlines():
    line1  = line.strip().split('\t')
    if line1[0] not in sku:
      if len(dic1)>0:
        d = sorted(dic1.items(), key=lambda d: d[1],reverse=True)
        for d1,d2 in d:
          if d2>=0:
            w1.write(d1+'\t'+str(d2)+'\n')
      sku[line1[0]]=1
      dic1={}
    if len(line1)>1:
      have=[]
      for key in dic:
        if line1[-1].find(key)>=0:
          have.append(key)
      have1 = set(have)
      len_o =len(line1[-1])
      len_=0
      for k in have1:
        len_+=len(k)
      if len_o==3 and len_/len_o<1:
        dic1[line.strip()]  = 0
      else:
        dic1[line.strip()]  = len_/len_o
  w1.flush()
#find_sale(sys.argv[5],sys.argv[9])
salepoint_len = sorted(dic.items(),key = lambda item:len(item[0]), reverse=True)
#print(salepoint_len)
r2 =open(sys.argv[9])
train_text=''
text_list=[]
for line in r8.readlines():
   line1  = line.strip()
   text_list.append(line1)
   train_text+=line1

def read_idf(filename):
   if '/' in filename:
    filename = filename.replace('/','_')
   filename = filename+'_idf'
   dic_idf ={}
   dic_idf1={}
   r = open(os.path.join(root,filename),'r')
   for line in r.readlines():
     line1 = line.strip().split('\t')
     if len(line1)==3:
       dic_idf[line1[0]]=float(line1[1])
       dic_idf1[line1[0]]=float(line1[2])
   return dic_idf1,dic_idf

def cal_sim(list_list1,list2):
   #print(list1)
   #print(list2)
   for list_1 in list_list1:
     num=0
     for c in list2:
       if c in list_1:
         num+=1
         if num>1:
           return True
   return False

def uniq(list_sku):
   writen=[]
   writenn=[]
   for list_line in list_sku:
     line = list_line.strip().split('\t')[0]
     cur_list = jieba.lcut(line)
     cur_set = list(set(cur_list))
     if len(writen)==0:
       writen.append(cur_set)
       writenn.append(list_line)
       continue
     if not cal_sim(writen,cur_set):
       writen.append(cur_set)
       writenn.append(list_line)
   return writenn
skus={} #存挖出卖点的sku:对应卖点
tmp=[]
sku=''
ssku=[]
import re
def contain(sent):
   my_re = re.compile(r'[A-Za-z]')
   my_re1 = re.compile(r'[0-9]+')
   if bool(re.findall(my_re1, sent)):
      return True
   return False
nummm=0

name=[]
cat=[]
def not_name_cat(salepoint):
   for name in name_title:
      if salepoint.find(name)>=0:
           return False
   for dic_t in cat_title:
      if salepoint.find(dic_t)>=0:
           return False
   for words in salepoint:
       if '\u4e00' <=  words  <= '\u9fff':
           return True
   return False

def not_chi(salepoint, k):
   index = salepoint.find(k)
   if index<1:
      return False
   for i in range(0,max(2,index)):
      if '\u4e00' <= salepoint[i]  <= '\u9fff':
          return False
   return True
import collections
#def expand_sp(salepoint):
#    tmp_l={}
#    c_l=0
#    tmp_r={}
#    c_r=0
#    num=1
#    for text in text_list:
#      if text.find(salepoint)>=0:
#        index = text.find(salepoint)
#        index_r = text.find(salepoint) + len(salepoint)
#        left = index-1
#        right = index_r
#        l=''
#        r=''
#        while left>=max(0,index-num):
#          l=text[left]+l
#          if l not in tmp_l:
#            tmp_l[l]=1
#          else:
#            tmp_l[l]+=1
#          c_l+=1
#          left-=1
#        while right<=min(len(text)-1, index_r+num-1):
#          r+=text[right]
#          if r not in tmp_r:
#            tmp_r[r]=1
#          else:
#            tmp_r[r]+=1
#          c_r+=1
#          right+=1
   # if salepoint.find('触摸显示')>=0:
      #(collections.Counter(tmp_l))
    
      #print(collections.Counter(tmp_r))
#    d_l = sorted(tmp_l.items(), key=lambda d: d[1],reverse=True)
#    d_r = sorted(tmp_r.items(), key=lambda d: d[1],reverse=True)
#    sp = salepoint
#    if len(d_l)>0:
#      k_l,v_l = d_l[0]
#      if float(v_l)/c_l >0.8 and k_l not in ['，','。']:
#        sp = k_l+salepoint
#        print(salepoint)
#        print(sp)
#        exit(0)
#    if len(d_r)>0:
#      k_r,v_r = d_r[0]
#      if float(v_r)/c_r >0.8 and k_r not in ['，','。']:
#        sp = sp+k_r 
#        print(salepoint)
#        print(sp)
#        exit(0)
#    if salepoint=='触摸显示':
#       print('111111')
#       print(sp)
#       print('222222')
#    if salepoint=='循环风量':
#       print('111111')
#       print(sp)
#       print('222222')
#    return sp
def clean_sp(salepoint, all_title_sp, all_title):
    for title in all_title:
      if title.find(salepoint)>=0 and not_name_cat(salepoint):
        return salepoint
    for k,v in salepoint_len:
      if salepoint.find(k)>=0 and len(k)>2:
         if not_name_cat(k):
           if not_chi(salepoint, k):
             tmp_re=(salepoint[0:salepoint.find(k)]+k)
             return tmp_re
           else:
             return k
    return '' 
def isOkInput(sents):
  if len(sents)<2:
     return False
  for i in range(len(sents)):
    if '\u4e00' <= sents[i] <= '\u9fff': 
       return True
  return False
for line in r2.readlines():
   if nummm%100==0:
      print(str(nummm))
   nummm+=1
   if line.strip().split('\t')[0] not in skus:
      skus[line.strip().split('\t')[0]]=[]
   if train_text.find(line.strip().split('\t')[-2])>=0 and isOkInput(line.strip().split('\t')[-2]):
      skus[line.strip().split('\t')[0]].append(line.strip().split('\t')[-2])
skuss=[] #所有需要输出的sku
for line in r111.readlines():
   line1 = line.strip()
   skuss.append(line1)

table_text=[] #所有需要输出的sku的对用table
for line in r11.readlines():
  line1 = line.strip()
  table_text.append(line1)

l_num=0
for j in table_text:
  sku_n = skuss[l_num]
  l_num+=1
  js = j.split('##')
  s = js[0]
  ss = s.split('$$')
  key=ss[0].strip()
  values = ss[1].strip()
  values = re.sub('\W', '', values)
  #w1.write(str(l_num)+'\n')
  w2.write('https://item.jd.com/%s.html' % sku_n+'\n')
  w2.write(values+'\n')
  w2.write('\n')
w2.flush()
r1 = open(sys.argv[4],'r')
texts=[]
tmp=[]
for line in r1.readlines():
  if line=='\n':
    texts.append(tmp)
    tmp=[]
  else:
    tmp.append(line.strip())
skuus={}
assert(len(skuss)==len(table_text))
for i in range(len(skuss)):
  skuus[skuss[i]] = table_text[i].split('CategoryL3 $$')[1].split('##')[0].strip()
  print(table_text[i].split('CategoryL3 $$')[1].split('##')[0].strip())

def read_nle(filename):
  num=0
  nle1={}
  if '/' in filename:
    filename = filename.replace('/','_')
  filename = filename+'_nle'
  r = open(os.path.join(root,filename),'r')
  for line in r.readlines():
    if line =='\n':
      num=0
      continue
    elif num==0:
      if line.strip() not in nle1:
        nle1[line.strip()]=[]
      name=line.strip()
      num+=1
      continue
    elif num==1:
      num+=1
      continue
    elif num==2:
      lines = line.strip().split('\t')
      for s in lines:
        ss =s.split(':')
        for ks in ss:
          nle1[name].append(ks)
  return nle1  

def haveChi(check_str):
   for ch in check_str:
      if u'\u4e00' <= ch <= u'\u9fff':
          return True
   return False
F = False
for text in texts:
  sku=''
  words=[]
  for index in range(len(text)):
     if index==0:
        if F == True:
          w1.write('\n')
          F = False
        sku = text[index].split('https://item.jd.com/')[1].split('.html')[0] 
        w1.write(text[index]+'\n')
     if index==1:
        w1.write(text[index]+'\n')
        F = True
     else:
        word = text[index].split('\t')[0]
        words.append(word)
  if sku in skus:
    sku_item = sku
    www=[]
    sales = skus[sku]
    for sale in sales:
      #for w in words:
       # if sale.find(w)>=0:
      if sale.find('/')<0 and sale.find(':')<0 and sale.find('?')<0 and sale.find('？')<0 and sale.find('：')<0 and haveChi(sale)and sale.find(',')<0 and sale.find('，')<0 and sale.find('。')<0 and sale.find('、')<0:   
        www.append(sale)
        #  break
      #if contain(sale):
       #  www.append(sale)
    #w_set = set(www)
    wwww = set(www)
    rank={}
    for s in wwww:
     importen={}
     nle={}
     word_dict_idf={}
     list_sub_sen = jieba.lcut(s)
     cat = skuus[sku_item]
     try:
       nle = read_nle(cat)
     except:
       w1.write('\n')
       continue
     _,word_dict_idf = read_idf(cat)
     
     for ke in nle:
       sor =0
       tttmp=[]
       for k in list_sub_sen:
         if k in nle[ke]:
           tttmp.append(k)
           try:
             sor += float(nle[ke][nle[ke].index(k)+1])
           except:
             continue
       importen[ke]=sor/len(list_sub_sen)
     d1 = sorted(importen.items(), key = lambda x: x[1], reverse=True)
     if len(d1)==0:
       continue
     sore = 0
     for kk in list_sub_sen:
       if kk in word_dict_idf:
         sore+=word_dict_idf[kk]
     sore = sore/len(list_sub_sen)
     rr = 'other'
     try:
       if d1[0][1]-d1[1][1]< 0.001:
         rr = 'other'
       else:
         rr = d1[0][0]
       if rr.find('Brand')==0:
         rr = 'other'
     except:
       pass
     rank[s+'\t'+rr] = float(sore)
     #rank[s+'\t'+rr] = float(d1[0][1])
    d2 = sorted(rank.items(), key = lambda x: x[1], reverse=True)
    houxuan = []
    for d21,d22 in d2:
       houxuan.append(d21+'\t'+str(d22))
    s1 = uniq(houxuan)
    w_dic={}
    for s11 in s1:
      if s11.split('\t')[1] not in w_dic:
        w_dic[s11.split('\t')[1]] = []
      w_dic[s11.split('\t')[1]].append(s11.split('\t')[0]+':'+'%.5f' % float(s11.split('\t')[2]))
    for w in w_dic:
      w1.write(w+'\t'+'\t'.join(w_dic[w])+'\n')
    w1.write('\n')
    F =False 
w1.write('\n') 
