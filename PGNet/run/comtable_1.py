import sys
import os
r1 = open(sys.argv[1],'r')
r2 = open(os.path.join(sys.argv[2],'train.text'),'r')
w1 = open(sys.argv[1]+"_out",'w')
dic1={}
total_text=''
for line in r2.readlines():
  line1 = line.strip()
  total_text+=line1
num=0
for line in r1.readlines():
   num+=1
   if num%1000==0:
     print(num)
   line1 = line.strip().split('\t')
   s  = line1[-1].replace('##',' ').replace('$$',' ')
   line1[-1] = s
   w1.write('\t'.join(line1)+'\n')
   
   #if total_text.find(line1[-1])>=0:
   #  w1.write(line)     
