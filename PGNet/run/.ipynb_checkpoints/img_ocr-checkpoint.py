import json
import time
import requests
import sys
r1 = open(sys.argv[1],'r')
w1 = open(sys.argv[1]+'_out','w')
w2 = open(sys.argv[1]+'_error','w')
num=0
skus=[]
bad_res=[]
for line in r1.readlines():
    if num%100==0:
        print(num)
    num+=1
    try:
        line1 = line.strip().split('\t')
        url = line1[2]
        sku = line1[0]
        p_id = line1[1]
        url = url.strip()
        TOKEN = "0c7bada0-cede-45a3-84d2-76052d3d3182"
        img_url = url.strip()
        url = "http://g.jsf.jd.local/com.jd.cv.jsf.ComputerVisionService/OCR-GPU/recognizeCharacter/40431/jsf/29000"
        data = [TOKEN, img_url, {"imgusecache": "true", "isusecache": "true"}]
        res = json.loads(requests.post(url, json.dumps(data)).text)
        ocr=[]
        dic={}
        if 'texts' in res:
            ocr = res['texts']
            dic[sku] = ocr
            d = json.dump(dic,w1,ensure_ascii=False)
            w1.write('\n')
        else:
            bad_res.append(line)
    except Exception as e:
         
        bad_res.append(line)
        pass
time = 0
while(time<4):
  for line in bad_res:
    try:
      line1 = line.strip().split('\t')
      url = line1[2]
      sku = line1[0]
      p_id = line1[1]
      
      TOKEN = "0c7bada0-cede-45a3-84d2-76052d3d3182"
      img_url = url.strip()
      url = "http://g.jsf.jd.local/com.jd.cv.jsf.ComputerVisionService/OCR-GPU/recognizeCharacter/40431/jsf/29000"
      data = [TOKEN, img_url, {"imgusecache": "true", "isusecache": "true"}]
      res = json.loads(requests.post(url, json.dumps(data)).text)
      ocr=[]
      dic={}
      if 'texts' in res:
        ocr = res['texts']
        dic[sku] = ocr
        d = json.dump(dic,w1,ensure_ascii=False)
        w1.write('\n')
        bad_res.remove(line)
    except Exception as e:

        pass 
  time+=1
w1.flush()

r1.close()
w1.close()
w2.close()
