import sys
import functools
import math
import re
import os

class Pos():
    def __init__(self, start, end):
        self.start = start
        self.end = end


def combin_array(steps, new_steps, pointer, target):
    if pointer + 1 == len(steps):
        new_steps.append(target)
        return
    next_ = steps[pointer + 1]
    if (next_.start <= target.end):
        target.end = max(next_.end, target.end)
        combin_array(steps, new_steps, pointer + 1, target)
    else:
        new_steps.append(target)
        combin_array(steps, new_steps, pointer + 1, next_)


def highlight_processer(text, sps):
    steps = []
    for sp in sps:
        sp = sp.strip()
        pos = text.find(sp)
        while (pos > -1):
            pos_ = Pos(pos, pos + len(sp))
            steps.append(pos_)
            pos = text.find(sp, pos + len(sp))
    if len(steps) == 0:
        return text

    def cmp(a, b):
        if a.start > b.start:
            return 1
        elif a.start < b.start:
            return -1
        else:
            return 0

    res = sorted(steps, key=functools.cmp_to_key(cmp))
    newSteps = []
    combin_array(res, newSteps, 0, res[0])
    result = []
    begin = 0
    for i in newSteps:
        result.append(text[begin: i.start])
        result.append("<strong>")
        result.append(text[i.start: i.end])
        result.append("</strong>")
        begin = i.end
    result.append(text[begin: len(text)])
    return ''.join(result)

def postprocess(res):
    res_ = ""
    res_final = []
    # if res.find('</strong>') < 0:
    #     return res
    for word_index in range(len(res)):
        if res[word_index] in ['；', '。', '！']:
            if res_.find('</strong>') < 0:
                res_final.append(res_)
                res_final.append(res[word_index])
            else:
                index_end = res_.index("</strong>")
                res_tmp = res_[0: index_end + len("</strong>")]
                second_sent = res_[index_end + len("</strong>"): ]
                second_sent += res[word_index]
                second_sent = second_sent.replace("</strong>", "")
                second_sent = second_sent.replace("<strong>", "")
                res_final.append(res_tmp)
                res_final.append(second_sent)
            res_ = ""
        else:
            res_+=res[word_index]

    return ''.join(res_final)


def highlight(text_file, sp_file, out_file):
    w1 = open(out_file, 'w')
    r1 = open(text_file, 'r')
    r2 = open(sp_file, 'r')
    sp = r2.readlines()
    for text in r1.readlines():
        text = text.strip().split('\t')[2]
        text = text.strip().replace(' ', '')
        res = highlight_processer(text, sp)
        res = postprocess(res)
        w1.write(res+'\n')
    w1.flush()


if __name__ == "__main__":
    path = sys.argv[1]  #"text"
    domain = sys.argv[2]    #"shuma_sp_0"
    xuanpin = sys.argv[3]   #"out.txt"
    sp_file = os.path.join(path, 'data', domain+'_rawData', 'title_second_sp')
    root_path = os.path.join(path, 'data', domain+'_'+xuanpin)
    text_file = os.path.join(root_path, 'res')
    out_file = os.path.join(root_path, 'high_light_res')
    highlight(text_file, sp_file, out_file)
