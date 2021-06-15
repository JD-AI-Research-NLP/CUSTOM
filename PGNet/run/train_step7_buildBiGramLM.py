import sys
import pickle
import jieba
import os

root = sys.argv[1]
domain = sys.argv[2]
folder = sys.argv[3]

inputFile = os.path.join(root, 'data', domain, 'train.text')
out1File = os.path.join(root, 'data', 'biLM1.pkl')
out2File = os.path.join(root, 'data', 'biLM2.pkl')

dic1 = {}
dic2 = {}

with open(inputFile, 'r', encoding='utf-8') as f:
    for line in f:
        tokens = ['[START]'] + [_.strip() for _ in jieba.cut(line.lower().strip()) if _.strip() != ''] + ['[STOP]']
        for i in range(len(tokens) - 1):
            tok = tokens[i]
            nextTok = tokens[i + 1]
            if tok not in dic2:
                dic2[tok] = {}
                dic1[tok] = 0
            dic1[tok] += 1
            if nextTok not in dic2[tok]:
                dic2[tok][nextTok] = 0
            dic2[tok][nextTok] += 1

with open(out1File, 'wb') as f1, open(out2File, 'wb') as f2:
    pickle.dump(dic1, f1)
    pickle.dump(dic2, f2)