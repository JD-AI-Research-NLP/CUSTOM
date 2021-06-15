import sys
import os
import jieba
root = sys.argv[1]
domain = sys.argv[2]
os.mkdir(os.path.join(root, 'data', 'trigram'))
os.mkdir(os.path.join(root, 'data', 'trigram', domain))
f1 = os.path.join(root, 'data', domain, 'train.text')
f2 = os.path.join(root, 'data', 'trigram', domain, 'triletter')
f3 = os.path.join(root, 'data', 'trigram', domain, 'trigram')
f4 = os.path.join(root, 'data', 'trigram', domain, 'train_dic')
f5 = os.path.join(root, 'data', 'trigram', domain, 'source.dict')
f6 = os.path.join(root, 'data', 'trigram', domain, 'source.dict.copy')
r1 = open(f1,'r')
w1 = open(f2,'w')
w2 = open(f3,'w')
w3 = open(f4,'w')
w4 = open(f5,'w')
w5 = open(f6,'w')
ngram={}
tmp = []
for line in r1.readlines():
  line1 = line.strip()
  sents_len = len(line1)
  for i in range(0,sents_len-3):
    temp=''
    temp+=line1[i]
    temp+=line1[i+1]
    temp+=line1[i+2]
    if temp not in ngram:
      ngram[temp]=1
    else:
      ngram[temp]+=1
for i in ngram:
  w1.write(i+'\t'+str(ngram[i])+'\n')
r2 = open(f1,'r')
ngram={}
for line in r2.readlines():
  line1 = jieba.lcut(line.strip().lower())
  sents_len = len(line1)
  for i in range(0,sents_len-3):
    temp=[]
    temp.append(line1[i])
    temp.append(line1[i+1])
    temp.append(line1[i+2])
    if ' '.join(temp) not in ngram:
      ngram[' '.join(temp)]=1
    else:
      ngram[' '.join(temp)]+=1
for i in ngram:
  w2.write(i+'\t'+str(ngram[i])+'\n')
