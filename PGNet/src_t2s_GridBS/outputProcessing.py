# -*- coding: UTF-8 -*-
def expendOutput(prediction_file, count_file):
  with open(prediction_file, 'r', encoding='utf-8') as sr_p:
    with open(count_file, 'r', encoding='utf-8') as sr_c:
      with open(prediction_file + ".expend", 'w', encoding='utf-8') as sw:
        for line in sr_p:
          cnt = int(sr_c.readline().strip())
          for i in range(cnt):
            sw.write(line)
            
def cutTerminator(file):
  with open(file, 'r', encoding='utf-8') as f:
    newLines = [_.strip('\n 。') for _ in f]
    with open(file + '.processed', 'w', encoding='utf-8') as fw:
      for line in newLines:
        fw.write(line + '\n')
        
if __name__ == "__main__":
  import sys
  expendOutput(sys.argv[1],sys.argv[2])
  #cutTerminator(sys.argv[1])