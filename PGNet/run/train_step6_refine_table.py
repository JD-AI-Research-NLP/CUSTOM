import os
import sys
def handle(path, mode):
  r1 = open(os.path.join(out_data_path, mode+'.table.com'),'r')
  r2 = open(os.path.join(out_data_path, mode+'.sku'),'r')
  r3 = open(os.path.join(out_data_path, mode+'.text'),'r')
  r4 = open(os.path.join(out_data_path, mode+'.title'),'r')
  r5 = open(os.path.join(out_data_path, mode+'.id'),'r')
  r6 = open(os.path.join(out_data_path, mode+'.table'),'r')
  if mode=="train":
    r7 = open(os.path.join(out_data_path, mode+'.fakeValidOcr'),'r')
    r8 = open(os.path.join(out_data_path, mode+'.fakeConstraint'),'r')
  else:
    r7 = open(os.path.join(out_data_path, mode+'.validSku.flag'),'r')
    r8 = open(os.path.join(out_data_path, mode+'.fakewriting'),'r')  


  
  w1 = open(os.path.join(out_data_path, mode+'.table.com.tmp'),'w')
  w2 = open(os.path.join(out_data_path, mode+'.sku.tmp'),'w')
  w3 = open(os.path.join(out_data_path, mode+'.text.tmp'),'w')
  w4 = open(os.path.join(out_data_path, mode+'.title.tmp'),'w')
  w5 = open(os.path.join(out_data_path, mode+'.id.tmp'),'w')
  w6 = open(os.path.join(out_data_path, mode+'.table.tmp'),'w')
  if mode=="train":
    w7 = open(os.path.join(out_data_path, mode+'.fakeValidOcr.tmp'),'w')
    w8 = open(os.path.join(out_data_path, mode+'.fakeConstraint.tmp'),'w')
  else:
    w7 = open(os.path.join(out_data_path, mode+'.validSku.flag.tmp'),'w')
    w8 = open(os.path.join(out_data_path, mode+'.fakewriting.tmp'),'w')
  
  com = r1.readlines()
  sku = r2.readlines()
  text = r3.readlines()
  title = r4.readlines()
  id_ = r5.readlines()
  table = r6.readlines()
  if mode=="train":
    t1 = r7.readlines()
    t2 = r8.readlines()
  else:
    t1 = r7.readlines()
    t2 = r8.readlines()

  assert(len(t1)==len(t2)==len(com)==len(sku)==len(text)==len(title)==len(id_)==len(table))
  for i in range(len(com)):
     if com[i].split(' ## ')[0].find('。')<5:
        continue
     else:
        w1.write(com[i])
        w2.write(sku[i])
        w3.write(text[i])
        w4.write(title[i])
        w5.write(id_[i])
        w6.write(table[i])
        w7.write(t1[i])
        w8.write(t2[i])
  w1.flush()
  w2.flush()
  w3.flush()
  w4.flush()
  w5.flush()
  w6.flush()
  w7.flush()
  w8.flush()
 
  os.replace(os.path.join(out_data_path,mode+'.table.com.tmp'), os.path.join(out_data_path,mode+'.table.com'))
  os.replace(os.path.join(out_data_path,mode+'.sku.tmp'), os.path.join(out_data_path, mode+'.sku'))
  os.replace(os.path.join(out_data_path,mode+'.text.tmp'),os.path.join(out_data_path,mode+'.text'))
  os.replace(os.path.join(out_data_path,mode+'.title.tmp'),os.path.join(out_data_path,mode+'.title'))
  os.replace(os.path.join(out_data_path,mode+'.id.tmp'),os.path.join(out_data_path,mode+'.id'))
  os.replace(os.path.join(out_data_path,mode+'.table.tmp'),os.path.join(out_data_path,mode+'.table'))
  if mode=="train":
    os.replace(os.path.join(out_data_path,mode+'.fakeValidOcr.tmp'),os.path.join(out_data_path,mode+'.fakeValidOcr'))
    os.replace(os.path.join(out_data_path,mode+'.fakeConstraint.tmp'),os.path.join(out_data_path,mode+'.fakeConstraint'))
  else:
    os.replace(os.path.join(out_data_path,mode+'.validSku.flag.tmp'),os.path.join(out_data_path,mode+'.validSku.flag'))
    os.replace(os.path.join(out_data_path,mode+'.fakewriting.tmp'),os.path.join(out_data_path,mode+'.fakewriting'))

def reject_no_ocr(out_data_path):
  handle(out_data_path, "train")
  handle(out_data_path, "dev")  
if __name__ == "__main__":
  root = os.path.join(sys.argv[1], 'data')
  domain = sys.argv[2]
  dataset = sys.argv[3]
  folder = domain if dataset == "train" else domain + '_' + dataset
  in_sku_file = ".sku"
  out_data_path = os.path.join(root, folder)
  reject_no_ocr(out_data_path) 
