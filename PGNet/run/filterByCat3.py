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

def read_supported_cat3s(file):
    with open(file, 'r') as f:
        supportCat3s = [_.strip().split('\t')[1] for _ in f.readlines()]
    return supportCat3s

def valid_cat(path, domain, dataset):
    supportCat3s = read_supported_cat3s(os.path.join(path, 'data', domain+"_rawData", "thirdids_supported"))
    root = os.path.join(path, 'data', domain + '_' + dataset)

    infile = os.path.join(root, 'res')
    outfile = os.path.join(root, 'res.withUrl')
    with open(infile, 'r', encoding='utf-8') as f, open(outfile, 'w', encoding='utf-8') as w:
        for line in f:
            parts = line.split('\t')
            url = 'https://item.jd.com/' + parts[0] + '.html'
            title = parts[1]
            text = parts[2]
            w.write(url + '\t' + title + '\t' + text)
    
    infile = os.path.join(root, r'out1')
    outfile = os.path.join(root, r'cat3s')
    with open(infile, 'r', encoding='utf-8') as f, open(outfile, 'w', encoding='utf-8') as w:
        cnt = 0
        for line in f:
            line = line.strip()
            cat3 = [kv.split(' $$ ')[1] for kv in line.split(' ## ') if len(kv.split(' $$ ')) == 2 and kv.split(' $$ ')[0] == 'CategoryL3']
            if cat3[0] in supportCat3s:
                w.write('1' + '\t' + cat3[0] + '\t' + line + '\n')
            else:
                w.write('0' + '\t' + cat3[0]  + '\t' + line + '\n')
            cnt += 1
    
def merge_data(path, domain, dataset, sample_num):
    result = list()
    root = os.path.join(path, 'data', domain + "_" + dataset)
    print('reading data...')
    print(root)
    sku = read_data(os.path.join(root, dataset+'.validSku'))
    title = read_data(os.path.join(root, dataset+'.title'))
    text = read_data(os.path.join(root, dataset+'.prediction.writing'))
    cat3 = read_data(os.path.join(root, 'cat3s'))

    print('starting merge...')
    for index in range(len(sku)):
        if text[index] == 'NULL' or cat3[index].split('\t')[0]== '0':
            continue
        decoded = join_tokens(text[index].split())
#         decoded = ''.join(text[index].split())
        buf = sku[index] + '\t' + title[index] + '\t' +decoded
        result.append(buf)
    print('total sku number: ', len(text))
    print('valid sku number: ', len(result))
    print('product ratio: ', 1.0 - (len(text)-len(result))/len(text))
    print('saving')
    save_path = os.path.join(root, 'res.filterBySupportedCat3')
    with open(save_path, 'w') as f:
        for item in result:
            f.write(item+'\n')
            
    infile = os.path.join(root, 'res.filterBySupportedCat3')
    outfile = os.path.join(root, 'res.filterBySupportedCat3.withUrl')

    with open(infile, 'r', encoding='utf-8') as f, open(outfile, 'w', encoding='utf-8') as w:
        for line in f:
            parts = line.split('\t')
            url = 'https://item.jd.com/' + parts[0] + '.html'
            title = parts[1]
            text = parts[2]
            w.write(url + '\t' + title + '\t' + text)
    
if __name__ == "__main__":
    valid_cat(path, cate, dataset)
    merge_data(path, cate, dataset, sample_num)
