import os
import sys
import re
from random import sample
path = sys.argv[1]
cate = sys.argv[2]
sample_num = sys.argv[3]
dataset = sys.argv[4]
def read_data(path):
    result = list()
    with open(path, 'r') as f:
        for line in f:
            result.append(line.strip())
    return result

def join_tokens(tokens):
    pattern = re.compile(r'[^\u4e00-\u9fa5，。？；！、]+$')
    tmp = ''
    for i in range(len(tokens)):
        if i > 0 and re.match(pattern, tokens[i-1] + ' ' + tokens[i]):
            tmp = tmp + ' ' + tokens[i]
        else:
            tmp = tmp + tokens[i]
    return tmp.strip()

def merge_data(path, xuanpin, dataset, sample_num):
    result = list()
    root = os.path.join(path, 'data', xuanpin + "_" + dataset)
    print('reading data...')
    print(root)
    sku = read_data(os.path.join(root, dataset+'.validSku'))
    title = read_data(os.path.join(root, dataset+'.title'))
    text = read_data(os.path.join(root, dataset+'.prediction.writing'))

    print('starting merge...')
    for index in range(len(sku)):
        if text[index] == 'NULL' or 'NULL' in title[index]:
            continue
        decoded = join_tokens(text[index].split())
#         decoded = ''.join(text[index].split())
        buf = sku[index] + '\t' + title[index] + '\t' +decoded
        result.append(buf)
    print('total sku number: ', len(text))
    print('valid sku number: ', len(result))
    print('product ratio: ', 1.0 - (len(text)-len(result))/len(text))
    print('saving')
    save_path = os.path.join(root, 'res')
    with open(save_path, 'w') as f:
        for item in result:
            f.write(item+'\n')
    print('sample: ', sample_num)
    try:
      sampled = sample(result, int(sample_num))
      sample_path = os.path.join(root, 'sample')
      with open(sample_path, 'w') as f:
         for item in sampled:
             f.write(item+'\n')
      print('finish')
    except:
      pass
    
if __name__ == "__main__":
    merge_data(path, cate, dataset,sample_num)
