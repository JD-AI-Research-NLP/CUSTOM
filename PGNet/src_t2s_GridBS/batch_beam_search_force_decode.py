# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to run beam search decoding"""

import jieba
import copy
import tensorflow as tf
import numpy as np
import data
from collections import defaultdict, OrderedDict
import re
import time
# from sortedcontainers import SortedListWithKey

FLAGS = tf.app.flags.FLAGS



class ConstraintHypothesis(object):
  """A (partial) hypothesis which maintains an additional constraint coverage object

  Args:
      token (unicode): the surface form of this hypothesis
      score (float): the score of this hypothesis (higher is better)
      coverage (list of lists): a representation of the area of the constraints covered by this hypothesis
      constraints (list of lists): the constraints that may be used with this hypothesis
      payload (:obj:): additional data that comes with this hypothesis. Functions may
          require certain data to be present in the payload, such as the previous states, glimpses, etc...
      backpointer (:obj:`ConstraintHypothesis`): a pointer to the hypothesis object which generated this one
      constraint_index (tuple): if this hyp is part of a constraint, the index into `self.constraints` which
          is covered by this hyp `(constraint_idx, token_idx)`
      unfinished_constraint (bool): a flag which indicates whether this hyp is inside an unfinished constraint

  """

  def __init__(self, tokens, log_probs, state, attn_dists, p_gens, s2s_coverage,
               coverage, constraints, payload=None, backpointer=None,
               constraint_index=None, unfinished_constraint=False, punctuation_split=True):

    assert len(coverage) == len(constraints), 'constraints and coverage length must match'
    assert all(len(cov) == len(cons) for cov, cons in zip(coverage, constraints)), \
      'each coverage and constraint vector must match'

    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.attn_dists = attn_dists
    self.p_gens = p_gens
    self.coverage = coverage
    self.s2s_coverage = s2s_coverage
    self.constraints = constraints
    self.backpointer = backpointer
    self.payload = payload
    self.constraint_index = constraint_index
    self.unfinished_constraint = unfinished_constraint
    self.punctuation_split = punctuation_split
    self.s = 0
    self.word_dict={}
  def __str__(self):
    return u'token: {}, sequence: {}, score: {}, coverage: {}, constraints: {},'.format(
      self.token, self.sequence, self.score, self.coverage, self.constraints)

  def __getitem__(self, key):
    return getattr(self, key)

  def extend(self, token, log_prob, state, attn_dist, p_gen, s2s_coverage,
             coverage, constraints, payload=None, backpointer=None,
             constraint_index=None, unfinished_constraint=False, punctuation_split=True):
    """Return a NEW hypothesis, extended with the information from the latest step of beam search.

    Args:
      token: Integer. Latest token produced by beam search.
      log_prob: Float. Log prob of the latest token.
      state: Current decoder state, a LSTMStateTuple.
      attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
      p_gen: Generation probability on latest step. Float.
      coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
    Returns:
      New Hypothesis for next step.
    """
    return ConstraintHypothesis(tokens=self.tokens + [token],
                                log_probs=self.log_probs + [log_prob],
                                state=state,
                                attn_dists=self.attn_dists + [attn_dist],
                                p_gens=self.p_gens + [p_gen],
                                s2s_coverage=s2s_coverage,
                                coverage=coverage,
                                constraints=constraints,
                                payload=payload,
                                backpointer=backpointer,
                                constraint_index=constraint_index,
                                unfinished_constraint=unfinished_constraint,
                                punctuation_split=punctuation_split
                                )

  @property
  def latest_token(self):
    return self.tokens[-1]

  # def char_len(self, vocab, batch):
  #   #   return len(self.plain_text(vocab, batch))

  def plain_text(self, vocab, batch):
    output_ids = [int(t) for t in self.tokens[1:]]
    decoded_words = data.outputids2words(output_ids, vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))
    pred = ''.join(decoded_words).replace(' ', '').replace('\t', '').replace(data.STOP_DECODING, '')
    return pred

  def plain_word(self, vocab, batch):
    output_ids = [int(t) for t in self.tokens[1:]]
    decoded_words = data.outputids2words(output_ids, vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))
    return decoded_words

  @property
  def log_prob(self):
    # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
    return sum(self.log_probs)

  @property
  def avg_log_prob(self):
    # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
    return self.log_prob / len(self.tokens)

  def trigram_func_long(self,vocab,batch,trigram_v2):
    b = [a for a in self.__dict__.items()]
    text = self.plain_word(vocab,batch)
    txt=trigram_v2
    if len(text)<2:
      return 1
    else:
      start= 0
      end = len(text)
      index = end-2
      while(index>=0):
        if text[index]=='。':
           start = index+1
           index-=1
           break
        if text[index]=='，' and '，' in text[index+1:]:
           start = index+1
           index-=1
           break
        index-=1
      print(text)
      text=text[start:]
      if '。'==text[-1]:
        text = text[0:-1]
      if '[STOP]'==text[-1]:
        text=text[0:-2]
      print(text)
      if len(text)==0:
        return 1
      to_num =  len(text)
      cor_num=0
      for i in range(to_num):
        tri = ''.join(text[i:])
        if tri in txt:
           cor_num = to_num-i
           return float(cor_num)/to_num 
      return 0
  def trigram_func_final(self,vocab,batch,trigram_v2,dic):
    trigram = trigram_v2.getGram()
    text = self.plain_word(vocab,batch)
    if len(text)<3:
      return 1
    else:
      to_num =  len(text)-2
      cor_num=0
      for i in range(to_num):
        tri = ' '.join([text[i],text[i+1],text[i+2]]) 
        if tri in trigram:
           cor_num+=1
      return float(cor_num)/to_num  
  def trigram_func(self,vocab,batch,trigram_v2,dic):
    #b = [a for a in self.__dict__.items()]
    t1=time.time()
    sku_kb = batch.sku_kbs[0]
    cat3 = sku_kb['CategoryL3'.lower()][0].replace('/', '_')
    trigram1 = trigram_v2.getGram()
    pred_word = self.plain_word(vocab,batch)
    start_pred=0
    if len(pred_word)<3:
      return 1
    else:
      index=len(pred_word)-1
      start_pred=0
      end = len(pred_word)-1
      while(index>=0):
        if pred_word[index]=='，' or pred_word[index]=='、':
          end = index-1
          break
        if pred_word[index]=='。' or index<0:
          end=0
          start_pred = index+1
          trigram = trigram1
          break
        index-=1
      end_l=0
      index_pre=0
      index_l=end
      if end>0:
        while(index_l>0):
          if pred_word[index_l]=='，' or pred_word[index_l]=='；' or pred_word[index_l]=='！'or pred_word[index_l]=='。' or pred_word[index_l]=='、':
            end_l = index_l+1
            break
          index_l-=1
        tmp_sen = pred_word[end_l:end]
        tmp_str = ''.join(pred_word[end-1:end+1])
        start_pred = end-1
        if len(tmp_str)<4 and end-2>=0 and pred_word[end-2] not in ['；','！','，','。','、']:
          tmp_str = pred_word[end-2]+tmp_str
          start_pred = end-2
        if len(tmp_str)==0:
          tmp_str = ''.join(pred_word[end:end+1])
          start_pred = end
        if cat3+'_'+tmp_str not in dic:
          trigram= trigram1
        else:
          tmp_list = dic[cat3+"_"+tmp_str]
          tmp_dict_word={}
          for w_index in range(len(tmp_list)-2):
            tmp_dict_word[tmp_list[w_index]+' '+tmp_list[w_index+1]+' '+tmp_list[w_index+2]]=1
          trigram=tmp_dict_word
      else:
          trigram=trigram1
      t2 = time.time()
      if len(pred_word)-start_pred<3:
          return 1
      else:
          to_num =  len(pred_word)-start_pred-2
          cor_num=0
          for i in range(start_pred,len(pred_word)-2):
            tri = ' '.join([pred_word[i],pred_word[i+1],pred_word[i+2]])
            if tri in trigram:
              cor_num+=1
          return float(cor_num)/to_num
  @property
  def log_prob_v2(self):
    # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
    return min(self.log_probs)

  @property
  def avg_log_prob_v2(self):
    # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
    return self.log_prob_v2
  def contain_uniq_content_unigram(self, stopWords, vocab, batch):
    lastWord = data.outputids2words([self.tokens[-1]], vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))[0]
    previousWords = data.outputids2words(self.tokens[:-1], vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))
    if lastWord not in stopWords and lastWord in previousWords:
      return False
    return True

  def contain_uniq_bigrams(self):
    bigram_set = set()
    for i in range(len(self.tokens) - 1):
      bigram = str(self.tokens[i]) + "\t" + str(self.tokens[i + 1])
      if (bigram in bigram_set):
        return False
      bigram_set.add(bigram)
    return True

  def contain_no_redundant_continuous_tokens(self):
    for i in range(len(self.tokens) - 1):
      if self.tokens[i] == self.tokens[i + 1]:
        return False
    return True

  def contain_no_unk(self, vocab):
    if vocab.word2id(data.UNKNOWN_TOKEN) in self.tokens:
      return False
    return True

  def contain_no_badWrod(self, vocab, batch, black_list):
    pred = self.plain_text(vocab, batch)
    for word in black_list.getWords():
      if word in pred:
        return False
    return True

  def contain_uniq_triLetters(self, vocab, batch):
    triLetter_set = set()
    pred = self.plain_text(vocab, batch)
    if len(pred) <= 2:
      return True

    for i in range(len(pred) - 3, -1, -1):
      triLetter = str(pred[i]) + "\t" + str(pred[i + 1]) + "\t" + str(pred[i + 2])
      if (triLetter in triLetter_set):
        return False
      triLetter_set.add(triLetter)
    return True

  def not_start_with_specific_token(self, vocab, batch):
    pred = self.plain_text(vocab, batch)
    if pred.startswith('这款'):
      return False
    return True

  def have_exit_trigram(self, vocab, batch, trigram):
    punc_list=['，','。','、','；','？','！']
    pred_word = self.plain_word(vocab, batch)
    trigram = trigram.getGram()
    if len(pred_word)<3:
      return True
    num = 3
    last_index = len(pred_word)
    tmp = []
    for i in range(last_index-num, last_index):
      if pred_word[i] in punc_list:
        return True
      tmp.append(pred_word[i])
    assert len(tmp)==num
    words = ' '.join(tmp)
    if words in trigram:
       return True
    return False

  def have_func_exit(self, vocab, batch):
    def eng_and_dit(String):
      for w in String:
        if (('0' <= w <= '9') or ('a'<=w<='z') or ('A'<=w<='Z')) and String!='[STOP]' and String!='[UNK]':
          return True
      return False
    caption = ''.join(batch.caption_tokens_list[0])
    pred_word = self.plain_word(vocab, batch)
    if len(pred_word)>1 and (pred_word[-2]=='搭配' or pred_word[-2]=='支持' or pred_word[-2]=='通过' or pred_word[-2]=='搭载' or pred_word[-2]=='拥有' or pred_word[-2]=='采用' or pred_word[-2]=='内置'):
      if caption.find(pred_word[-1])<=0:
         return False
    if eng_and_dit(pred_word[-1]):
      if caption.find(pred_word[-1])<0:
         return False
    #if len(pred_word)>3 and pred_word[-2]=='，' and caption.find(pred_word[-3])>=0 and caption.find(pred_word[-4])>=0:
    #  if caption.find(pred_word[-1])>=0:
    #     return False
    return True
  def have_exit_trigram_letter_v2(self, vocab, batch, trigram_v2):
    punc_list=['，']
    pred_word = self.plain_word(vocab, batch)
    trigram = trigram_v2.getGram()
    if len(pred_word)<3:
        return True
    tmp=[]
    for i in range(len(pred_word)-3,len(pred_word)):
      tmp.append(pred_word[i])
    tmp1 = ' '.join(tmp)
    if tmp1 not in trigram:
      return False
    return True

  def have_exit_trigram_letter(self, vocab, batch, trigram):
    punc_list=['，','。','、','；','？','！']
    pred_word = self.plain_text(vocab, batch)
    #print(len(pred_word))
    trigram = trigram.getGram()
    #pirint(pred_word)
    if len(pred_word)<3:
      return True
    num = 3
    last_index = len(pred_word)
    tmp = ''
    for i in range(0, last_index-num+1):
       tmp+=pred_word[i]
       tmp+=pred_word[i+1]
       tmp+=pred_word[i+2]
       #print(i)
       #if (pred_word[i] in punc_list) or (pred_word[i+1] in punc_list) or (pred_word[i+2] in punc_list):
       #  tmp=''
       #  continue
       if (tmp not in trigram):
         return False
       #(tmp in trigram and trigram[tmp]<2) 
       tmp=''
    return True

  def have_dic(self, vocab, batch, train_text, word_dict):
    pred_word = self.plain_word(vocab, batch)
    sku_kb = batch.sku_kbs[0]
    cat3 = sku_kb['CategoryL3'.lower()][0].replace('/', '_')
    txt = ""
    if cat3 in word_dict:
       txt = word_dict[cat3]
    else:
       if cat3 in train_text:
         txt+='$'.join(train_text[cat3])
         word_dict[cat3] = txt
       else:
         return False
    if len(pred_word)<3 or '，'not in pred_word:
       return True
    else:
      index=len(pred_word)-1
      start_pred=0
      end = len(pred_word)-1
      while(index>=0):
        if pred_word[index]=='，' or pred_word[index]=='、':
          end = index-1
          break
        if pred_word[index]=='。' or pred_word[index]=='；' or pred_word[index]=='！'or index==0:
          return True
        index-=1
      end_l=0
      index_l=end
      while(index_l>0):
        if pred_word[index_l]=='，' or pred_word[index_l]=='；' or pred_word[index_l]=='！'or pred_word[index_l]=='。' or pred_word[index_l]=='、':
          end_l = index_l+1
          break
        index_l-=1
      #print(pred_word)
      tmp_sen = pred_word[end_l:end+1]
      #print(tmp_sen)
      tmp_str = ''.join(pred_word[end-1:end+1])
      #print(tmp_str)
      if len(tmp_str)<4 and end-2>=0 and pred_word[end-2] not in ['；','！','，','。','、'] and txt.find(pred_word[end-2]+tmp_str)>=0:
        tmp_str = pred_word[end-2]+tmp_str
      if len(tmp_str)==0:
        tmp_str = ''.join(pred_word[end:end+1])
      find_index=0
      dic_list=[]
      t1 = time.time()
      word_tmp=[]
      if cat3+'_'+tmp_str in word_dict and len(word_dict[cat3+'_'+tmp_str])>0:
        word_tmp = word_dict[cat3+'_'+tmp_str]
      else:
        while(find_index<len(txt)):
          if txt.find(tmp_str,find_index+1)>=0:
            find_index = txt.find(tmp_str,find_index+1)
            start_t = find_index-1
            end_t = find_index+1
            while(start_t>=0):
              if txt[start_t]!='，' and txt[start_t]!='。' and txt[start_t]!='、' and txt[start_t]!='；' and txt[start_t]!='！':
                start_t-=1
              else:
                break
            T = False
            while(end_t<=len(txt)-1):
              if txt[end_t]!='。' and txt[end_t]!='！' and txt[end_t]!='；' and  (txt[end_t]!='，' or  T==False):
                if txt[end_t]=='，':
                   T =True
                end_t+=1
              else:
                 break
            dic_list.append(txt[start_t+1:end_t])
            if len(dic_list)>1000:
              break
          else:
            break
        if len(dic_list)==0:
       #print(tmp_str)
         return False
    #print(dic_list)
        str_tmp = '$'.join(dic_list)
        word_tmp = jieba.lcut(str_tmp)
        word_dict[cat3+'_'+tmp_str]=word_tmp
      time2 = time.time()
      #word_dic = {}
      #for w in word_tmp:
      #  if w not in word_dic:
      #    word_dic[w]=1
      #  else:
      #    word_dic[w]+=1
      #word_dic['。']=100
      #word_dic['，']=100
      #if pred_word[-1] not in word_dic:
        #print(pred_word)
        #print(cat3+'_'+tmp_str)
      #  return False
      
      txt_word_dic = ' '.join(word_tmp)
      #if pred_word[-1] not in word_dic:
      if txt_word_dic.find(pred_word[-1])<0:
         return False

      #if word_dic[pred_word[-1]]<1:
        #print(pred_word)
        #print(tmp_str)
        #print(dic_list)
      #  return False
      #for t in tmp_sen:
      #   if t not in word_dic:
      #      return False
      return True
  def one_sent_notexit_same_gram(self, vocab, batch):
    pred_word = self.plain_text(vocab, batch)
    num_words = len(pred_word)
    punc_list=['。','；','？','！']
    #if pred_word[-1] not in punc_list:
    #  return True
    word_end_index = num_words-1
    last_punc_word = word_end_index
    pre_punc_word = 0
    index = word_end_index-1
    while(index>=0):
      if pred_word[index] in punc_list:
        pre_punc_word = index
        break
      index-=1
    tmp=[]
    for i in range(pre_punc_word+1, num_words-1):
       if i<num_words-1:
         if pred_word[i]+pred_word[i+1] not in tmp:
           tmp.append(pred_word[i]+pred_word[i+1])
         else:
           return False
    return True

  def long_sents_without_punc(self, vocab, batch, max_len=15, max_num_word=10, min_len=3, min_num_word=2):
    pred = self.plain_text(vocab, batch)
    pred_word = self.plain_word(vocab, batch)
    num_words = len(pred_word)
    punc_list=['，','。','、','；','？','！']
    #if pred[-1] not in punc_list:
    #   return True
    sents_end_index = len(pred)-1
    last_punc = sents_end_index
    pre_punc = 0
    index = sents_end_index-1
    while(index>=0):
      if pred[index] in punc_list:
        pre_punc = index
        break
      index-=1
    
    word_end_index = num_words-1
    last_punc_word = word_end_index
    pre_punc_word = 0
    index = word_end_index-1
    while(index>=0):
      if pred_word[index] in punc_list:
        pre_punc_word = index
        break
      index-=1
    if pred_word[-1]=='[STOP]':
      return True
    if (last_punc - pre_punc > max_len) or (last_punc_word - pre_punc_word > max_num_word):
      return False
    if (pred[-1] in punc_list) and (pred_word[-1]!='[STOP]') and ((last_punc - pre_punc < min_len)or(last_punc_word - pre_punc_word < min_num_word)):
      #print(pred)
      if pred[-3:]=='同时，':
        return True
      #print(pred[-1])
      #print(pred_word)
      return False
    if pred[-1] =='。' and pred[pre_punc]=='。':
      return False
    return True

  # def consistent_with_KB(self, vocab, batch, ckg):
  #   writing = self.plain_text(vocab, batch)
  #   sku_kb = batch.sku_kbs[0]
  #   blackAttrList = ['skuname', 'brandname_full', 'brandname_en', 'labels'
  #                                                                 'CategoryL1'.lower(), 'CategoryL2'.lower(),
  #                    'CategoryL3'.lower(),
  #                    '特色推荐', '特色功能', '产品特点', '产品特色',
  #                    '产品类型', '规格_水温调节范围', '主体_品牌']
  #   # blackAttrList = ['brandname_full', 'brandname_en', 'brandname_cn']
  #
  #   blackValueList = ['有', '无', '是', '否', '支持', '不支持', '其他', '其它']
  #
  #   cat3 = sku_kb['CategoryL3'.lower()][0]
  #   # print('sku third cat %s' %cat3)
  #   cat_ckg = ckg.get_cat_ckg(cat3)
  #   for catAttr, catValues in cat_ckg.items():
  #     if catAttr in sku_kb and catAttr not in blackAttrList:
  #       intersect = []
  #       for catValueSet in catValues:
  #         if any([catValue in blackValueList for catValue in catValueSet]):
  #           continue
  #         for v in catValueSet:
  #           if v in ''.join(sku_kb[catAttr]):
  #             intersect.append(v)
  #         # intersect = set(sku_kb[catAttr]).intersection(set(catValueSet))
  #       if len(intersect) == 0:
  #         continue
  #       for catValueSet in catValues:
  #         for catValue in catValueSet:
  #           # if any([catValue in catValueSet and catValue not in intersect and catValue in writing for catValue in catValueSet])
  #           if len(list(set(catValueSet).intersection(intersect))) == 0 and \
  #               catValue in writing and \
  #               '免去' not in writing and \
  #               '避免' not in writing and \
  #               '不用' not in writing:
  #             sku_value = "".join(sku_kb[catAttr])
  #             if (any([catValue in sku_v or sku_v in catValue for sku_v in intersect])):
  #               continue
  #             # if (any([v in writing for v in sku_kb[catAttr]])):
  #             if (any([v in writing for v in intersect])):
  #               continue
  #             # for k, v in sku_kb.items():
  #             #   print("%s ||| %s" % (k, ' | '.join(v)))
  #             # print('Not consistent: KB uses [%s ||| %s] but writing contains [%s]'% (catAttr, sku_value, catValue))
  #             return False
  #   return True
  def punc_ok(self, vocab, batch,train_text):
    writing = self.plain_text(vocab, batch)
    punc_list=['、','；','！','，','。']
    punc_list_end=['；','！','。']
    pre_index=0
    pre_punc=''
    if len(writing)<1:
      return True
    if writing[-1] in punc_list_end:
      index = len(writing)-2
      while(index>=0):
        if writing[index] in punc_list:
          pre_index = index
          pre_punc = writing[index]
          break
        index-=1
      if (pre_index<2) or (pre_punc in punc_list_end):
        return False
      find_index=0
      dic_list=[]
      if len(writing)<4:
        return False
      tmp_str=writing[-5:-1]
      txt = ""
      for w in train_text:
        txt +='$'.join(train_text[w])
      while(find_index<len(txt)):
        if txt.find(tmp_str,find_index+1)>=0:
          find_index = txt.find(tmp_str,find_index+1)
          str_next = txt[find_index+4]
          str_next1 = txt[find_index+1]
          str_next2 = txt[find_index+2]
          str_next3 = txt[find_index+3]
          if str_next in punc_list:
            t2 = time.time()
            return True
        else:
          return False
      return False
    else:
     return True
  def repeat_rule(self, vocab, batch):
    writing = self.plain_text(vocab, batch)
    sku_kb = batch.sku_kbs[0]
    ocr_shortSeqs = ''.join(batch.ocr_shortSeqs_list[0])
    kb_str=''
    for d in sku_kb:
      kb_str+=d
      kb_str+=''.join(sku_kb[d])
    caption = ''.join(batch.caption_tokens_list[0])
    caption+=kb_str
    caption+=ocr_shortSeqs
    #if len(writing)>4:
    #  print(writing)
    #  print(writing[-2:])
      #if writing[-2:]=='手工' and writing.find('手工')!=len(writing)-2: 
    return True
  def consistent_with_KB(self, vocab, batch, ckg):
    writing = self.plain_text(vocab, batch)
    sku_kb = batch.sku_kbs[0]
    blackAttrList = ['skuname', 'brandname_full', 'brandname_en', 'labels',
                     'CategoryL1'.lower(), 'CategoryL2'.lower(),
                     'CategoryL3'.lower(),
                     '特色推荐', '特色功能', '产品特点', '产品特色',
                     '产品类型', '规格_水温调节范围', '主体_品牌']
    # blackAttrList = ['brandname_full', 'brandname_en', 'brandname_cn']

    blackValueList = ['有', '无', '是', '否', '支持', '不支持', '其他', '其它']

    cat3 = sku_kb['CategoryL3'.lower()][0]
    # print('sku third cat %s' %cat3)
    cat_ckg = ckg.get_cat_ckg(cat3)
    for catAttr, catValues in cat_ckg.items():
      if catAttr in sku_kb and catAttr not in blackAttrList:
        intersect = []
        for catValueSet in catValues:
          if any([catValue in blackValueList for catValue in catValueSet]):
            continue
          for v in sku_kb[catAttr]:
            if v in catValueSet:
              intersect.append(v)
          # intersect = set(sku_kb[catAttr]).intersection(set(catValueSet))
        if len(intersect) == 0:
          continue
        for catValueSet in catValues:
          for catValue in catValueSet:
            # if any([catValue in catValueSet and catValue not in intersect and catValue in writing for catValue in catValueSet])
            if len(list(set(catValueSet).intersection(intersect))) == 0 and \
                catValue in writing and \
                '免去' not in writing and \
                '避免' not in writing and \
                '免除' not in writing and \
                '告别' not in writing and \
                '解除' not in writing and \
                '省去' not in writing and \
                '无需' not in writing and \
                '不用' not in writing:
              sku_value = "".join(sku_kb[catAttr])
              if (any([catValue in sku_v or sku_v in catValue for sku_v in intersect])):
                continue
              # if (any([v in writing for v in sku_kb[catAttr]])):
              if (any([v in writing for v in sku_kb[catAttr]])):
                continue
              # for k, v in sku_kb.items():
              #   print("%s ||| %s" % (k, ' | '.join(v)))
              # print('Not consistent: KB uses [%s ||| %s] but writing contains [%s]'% (catAttr, sku_value, catValue))
              return False
    return True
  def ori_text_find(self, vocab, batch, source):
    t1 = time.time()
    pred_word = self.plain_text(vocab, batch)
    sku_kb = batch.sku_kbs[0]
    kb_str=''
    for d in sku_kb:
      kb_str+=d
      kb_str+=''.join(sku_kb[d])
    ocr_shortSeqs = ''.join(batch.ocr_shortSeqs_list[0])
    caption = ''.join(batch.caption_tokens_list[0])
    caption+=kb_str
    caption+=ocr_shortSeqs
    dict_source = source.getDicts()
    for w in dict_source:
      if pred_word.find(w)>=0 and caption.find(w)<0:
        return False
    #cat3 = sku_kb['CategoryL3'.lower()][0].replace('/', '_')
    t2 = time.time()
    return True
  def have_exit_trigram_letter_v3(self,vocab, batch, trigram):
    sku_kb = batch.sku_kbs[0]
    caption = ''.join(batch.caption_tokens_list[0])
    cat3 = sku_kb['CategoryL3'.lower()][0].replace('/', '_')
    punc_list=['，','。','、','；','？','！']
    pred_word = self.plain_text(vocab, batch)
    #print(len(pred_word))
    trigram = trigram.getGram()
    #pirint(pred_word)
    if len(pred_word)<3:
      return True
    num = 3
    last_index = len(pred_word)
    tmp = ''
    for i in range(0, last_index-num+1):
       tmp+=pred_word[i]
       tmp+=pred_word[i+1]
       tmp+=pred_word[i+2]
       tmp1=tmp
       #print(i)
       #if (pred_word[i] in punc_list) or (pred_word[i+1] in punc_list) or (pred_word[i+2] in punc_list):
       #  tmp=''
       #  continue
       if (tmp1 not in trigram) and (caption.find(tmp)<=0):
         return False
       #(tmp in trigram and trigram[tmp]<2)
       tmp=''
    return True 
  def consistent(self, vocab, batch, ckg, funcWords):
    #pat = r'([\u4e00-\u9fa5]*)([A-z0-9/()+°\.\-%#@&℃]+)([\u4e00-\u9fa5]*)'
    pat = r'([\u4e00-\u9fa5]*)([A-z0-9/()+°\.\-%#@&℃³]+)([\u4e00-\u9fa5]*)'
    writing = self.plain_text(vocab, batch)
    sku_kb = batch.sku_kbs[0]
    ocr_shortSeqs = batch.ocr_shortSeqs_list[0]
    caption = ''.join(batch.caption_tokens_list[0])
    constraints = [''.join(cons_tokens) for cons_tokens in batch.constraints_token_list[0]]
    blackAttrList = ['skuname', 'labels']
    blackValueList = ['有', '无', '是', '否', '支持', '不支持', '其他', '其它']

    #cat3 = sku_kb['CategoryL3'.lower()][0]
    cat3 = sku_kb['CategoryL3'.lower()][0].replace('/', '_')
    sku_values = [v for vs in sku_kb.values() for v in vs if v not in blackValueList]
    # print('sku third cat %s' %cat3)
    cat_ckg = ckg.get_cat_ckg(cat3)
    for catAttr, catValues in cat_ckg.items():
      if catAttr not in blackAttrList:
        for catValueSet in catValues:
          for catValue in catValueSet:
            # print('catValue:{}'.format(catValue))
            if catValue in blackValueList:
              continue
            if any([catValue in cons for cons in constraints]):
              continue
            if catValue in writing \
                and catValue not in sku_values \
                and catValue not in caption \
                and all([catValue not in ocr for ocr in ocr_shortSeqs]):
              return False
    m = re.findall(pat, writing)
    if len(m) != 0:
      v = m[-1][1]
      if v not in sku_values \
          and v not in caption \
          and all([v not in ocr for ocr in ocr_shortSeqs]) \
          and all([v not in cons for cons in constraints]):
        return False

    cat_funcWords = funcWords.get_cat_funcWords(cat3)
    for cat_funcWord in cat_funcWords:
      if cat_funcWord in writing \
          and cat_funcWord not in sku_values \
          and cat_funcWord not in caption \
          and all([cat_funcWord not in ocr for ocr in ocr_shortSeqs]):
        return False
    return True

  def should_not_be_pruned(self, vocab, batch, ckg, funcWords, black_list, stopWords, trigram,train_text, dic, source_dict):
    if self.contain_uniq_bigrams() \
        and self.contain_no_redundant_continuous_tokens() \
        and self.contain_no_unk(vocab) \
        and self.contain_no_badWrod(vocab, batch, black_list) \
        and self.contain_uniq_triLetters(vocab, batch) \
        and self.one_sent_notexit_same_gram(vocab, batch)\
        and self.have_func_exit(vocab, batch)\
        and self.not_start_with_specific_token(vocab, batch) \
        and self.long_sents_without_punc(vocab, batch, max_len=50, max_num_word=50, min_len=4, min_num_word=2)\
        and self.have_exit_trigram_letter_v3(vocab, batch, trigram)\
        and self.contain_uniq_content_unigram(stopWords, vocab, batch)\
        and self.punc_ok(vocab, batch, train_text)\
        and self.ori_text_find(vocab, batch, source_dict):

      return True
    return False

#and self.have_dic(vocab, batch, train_text, dic)\
#and self.have_dic(vocab, batch, train_text, dic)
  @property
  def sequence(self):
    sequence = []
    current_hyp = self
    while current_hyp.backpointer is not None:
      sequence.append((current_hyp.token, current_hyp.constraint_index))
      current_hyp = current_hyp.backpointer
    sequence.append((current_hyp.token, current_hyp.constraint_index))
    return sequence[::-1]

  def constraint_candidates(self):
    available_constraints = []
    for idx in range(len(self.coverage)):
      if self.coverage[idx][0] == 0:
        available_constraints.append(idx)

    return available_constraints


class AbstractBeam():

  # def __init__(self, size):
  #   # note: here we assume bigger scores are better
  #   # self.hypotheses = SortedListWithKey(key=lambda x: -x['score'])
  #   self.hypotheses = SortedListWithKey(key=lambda x: x.avg_log_prob)
  #   self.size = size
  #
  # def add(self, hyp):
  #   self.hypotheses.add(hyp)
  #   if len(self.hypotheses) > self.size:
  #     assert len(self.hypotheses) == self.size + 1
  #     del self.hypotheses[-1]

  def __init__(self, size):
    # note: here we assume bigger scores are better
    # self.hypotheses = SortedListWithKey(key=lambda x: -x['score'])
    self.hypotheses = []
    self.size = size

  def add(self, hyp,vocab,batch,trigram_v2,v1,v2,dic):
    self.hypotheses.append(hyp)
    #self.hypotheses = sort_hyps(self.hypotheses)
    self.hypotheses = sort_hyps_v4(self.hypotheses,vocab,batch,trigram_v2,v1,v2,dic)
    if len(self.hypotheses) > self.size:
      assert len(self.hypotheses) == self.size + 1
      del self.hypotheses[-1]

  def __len__(self):
    return len(self.hypotheses)

  def __iter__(self):
    for hyp in self.hypotheses:
      yield hyp


class ConstrainedDecoder(object):
  def __init__(self, beam_implementation=AbstractBeam):
    self.beam_implementation = beam_implementation

  def search(self, sess, model, vocab, ckg, funcWords, black_list, stopWords, batch,
             start_hyp, grid_height, enc_states, max_source_len, beam_size, trigram, trigram_v2,train_text, dic, source_dict):
    search_grid = OrderedDict()
    start_beam = self.beam_implementation(size=1)
    v1 = model._hps.arg1
    v2 = model._hps.arg2
    start_beam.vocab=vocab
    start_beam.batch=batch
    start_beam.trigram_v2=trigram_v2
    trigram_=train_text.getGram()
    ori_text = trigram_ 
    start_beam.add(start_hyp,vocab,batch,trigram_v2,v1,v2,dic)
    search_grid[(0, 0)] = start_beam
    print('search_grid length %d' % len(search_grid[(0, 0)]))
    for i in range(1, max_source_len + 1):
      print('TIME: {}'.format(i))
      # j_start = max(i - (max_source_len - grid_height), 0)
      j_start = 0
      j_end = min(i, grid_height) + 1
      beams_in_i = j_end - j_start
      for j in range(j_start, j_end):
        new_beam = self.beam_implementation(size=beam_size)
        if (i - 1, j) in search_grid:
          generation_hyps = self.get_generation_hyps(
            sess=sess, model=model, vocab=vocab, batch=batch, beam=search_grid[(i - 1, j)],
            enc_states=enc_states, ckg=ckg, funcWords=funcWords, black_list=black_list, stopWords=stopWords, trigram=trigram, train_text=ori_text, dic=dic, source_dict= source_dict)
          # print('generation_hyps length %d'%len(generation_hyps))
          for hyp in generation_hyps:
            # print('hyp.latest_token: %s' % hyp.latest_token)
            # if hyp.should_not_be_pruned(vocab=vocab, batch=batch, ckg=ckg, black_list=black_list, stopWords=stopWords):
            new_beam.add(hyp,vocab,batch,trigram_v2,v1,v2,dic)
          # print('new_beam length %d' % len(new_beam))

        if (i - 1, j - 1) in search_grid:
          new_constraint_hyps = self.get_new_constraint_hyps(
            sess=sess, model=model, vocab=vocab, batch=batch, beam=search_grid[(i - 1, j - 1)], enc_states=enc_states)
          continued_constraint_hyps = self.get_continued_constraint_hyps(
            sess=sess, model=model, vocab=vocab, batch=batch, beam=search_grid[(i - 1, j - 1)], enc_states=enc_states)
          for hyp in new_constraint_hyps:
            if hyp.should_not_be_pruned(vocab=vocab, batch=batch, ckg=ckg, funcWords=funcWords, black_list=black_list, stopWords=stopWords, trigram=trigram, train_text=ori_text,dic=dic, source_dict=source_dict):
              new_beam.add(hyp,vocab,batch,trigram_v2,v1,v2,dic)
          for hyp in continued_constraint_hyps:
            if hyp.should_not_be_pruned(vocab=vocab, batch=batch, ckg=ckg, funcWords=funcWords, black_list=black_list, stopWords=stopWords, trigram=trigram, train_text=ori_text,dic=dic, source_dict=source_dict):
              new_beam.add(hyp,vocab,batch,trigram_v2,v1,v2,dic)

        search_grid[(i, j)] = new_beam
        #print('index: (%d,%d)--length:%d' % (i, j, len(new_beam)))
    return search_grid

  def get_generation_hyps(self, sess, model, vocab, batch, beam, enc_states, ckg, funcWords, black_list, stopWords, trigram, train_text, dic, source_dict):
    """return all hyps which are continuations of the hyps on this beam

    hyp_generation_func maps `(hyp) --> continuations`

    the coverage vector of the parent hyp is not modified in each child
    """

    continuations = (
    self.dumb_generate(sess=sess, model=model, vocab=vocab, batch=batch, hyp=hyp, enc_states=enc_states,
                       ckg=ckg, funcWords=funcWords, black_list=black_list, stopWords=stopWords, n_best=FLAGS.beam_size, trigram=trigram, train_text=train_text,dic=dic, source_dict=source_dict)
    for hyp in
    beam if
    not hyp.unfinished_constraint and hyp.latest_token != vocab.word2id(data.STOP_DECODING))
    # flatten
    return (new_hyp for hyp_list in continuations for new_hyp in hyp_list)

  def get_new_constraint_hyps(self, sess, model, vocab, batch, beam, enc_states):
    """return all hyps which start a new constraint from the hyps on this beam

    constraint_hyp_func maps `(hyp) --> continuations`

    the coverage vector of the parent hyp is modified in each child
    """

    continuations = (
      self.dumb_generate_from_constraints(sess=sess, model=model, batch=batch, hyp=hyp, enc_states=enc_states)
      for hyp in beam if
    not hyp.unfinished_constraint and hyp.punctuation_split and hyp.latest_token != vocab.word2id(data.STOP_DECODING))

    # flatten
    return (new_hyp for hyp_list in continuations for new_hyp in hyp_list)

  def get_continued_constraint_hyps(self, sess, model, vocab, batch, beam, enc_states):
    """return all hyps which continue the unfinished constraints on this beam

    constraint_hyp_func maps `(hyp, constraints) --> forced_continuations`

    the coverage vector of the parent hyp is modified in each child

    """

    continuations = (
      self.dumb_continue_unfinished_constraint(sess=sess, model=model, batch=batch, hyp=hyp, enc_states=enc_states)
      for hyp in beam if hyp.unfinished_constraint and hyp.latest_token != vocab.word2id(data.STOP_DECODING))

    return continuations

  def dumb_generate(self, sess, model, vocab, batch, hyp, enc_states, ckg, funcWords, black_list, stopWords, n_best, trigram,train_text,dic, source_dict):
    # make k_best random hyp objects
    latest_token = hyp.latest_token
    state = hyp.state
    prev_coverage = hyp.s2s_coverage
    constraint_token = vocab.word2id(data.PAD_TOKEN)
    (next_tokens, next_scores, const_probs, new_states, attn_dists, p_gens, new_coverage) \
      = model.decode_onestep_constraint(sess=sess,
                                        batch=batch,
                                        latest_tokens=[latest_token],
                                        constraint_tokens=[constraint_token],
                                        enc_states=enc_states,
                                        dec_init_states=[state],
                                        prev_coverage=prev_coverage)
    new_hyps = []
    i = 0
    valid_cnt = 0
    while (valid_cnt < n_best and i < 100):
      # for i in range(n_best):
      # print('generate log_prob(%d,%d): %.4f' %(0,i, next_scores[0, i]))
      tok = data.outputids2words([next_tokens[0, i]], vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))[0]
      punctuation_split = True if tok in ['，', '。', '；', '！', '、'] else hyp.punctuation_split
      new_hyp = hyp.extend(token=next_tokens[0, i],
                           log_prob=next_scores[0, i],
                           state=new_states[0],
                           attn_dist=attn_dists[0],
                           p_gen=p_gens[0],
                           s2s_coverage=new_coverage[0],
                           coverage=copy.deepcopy(hyp.coverage),
                           constraints=hyp.constraints,
                           payload=None,
                           backpointer=hyp,
                           constraint_index=None,
                           unfinished_constraint=False,
                           punctuation_split=punctuation_split
                           )
      if new_hyp.should_not_be_pruned(vocab=vocab, batch=batch, ckg=ckg, funcWords=funcWords, black_list=black_list, stopWords=stopWords, trigram=trigram,train_text=train_text,dic=dic ,source_dict=source_dict):
        new_hyps.append(new_hyp)
        valid_cnt += 1
      i += 1

    return new_hyps

  def dumb_generate_from_constraints(self, sess, model, batch, hyp, enc_states):
    """Look at the coverage of the hyp to get constraint candidates"""

    assert hyp.unfinished_constraint is not True, 'hyp must not be part of an unfinished constraint'
    new_constraint_hyps = []
    available_constraints = hyp.constraint_candidates()
    latest_token = hyp.latest_token
    last_state = hyp.state
    prev_coverage = hyp.s2s_coverage
    for idx in available_constraints:
      # starting a new constraint
      constraint_token = hyp.constraints[idx][0]
      # this should come from the model
      (next_tokens, next_scores, const_probs, new_states, attn_dists, p_gens, new_coverage) \
        = model.decode_onestep_constraint(sess=sess,
                                          batch=batch,
                                          latest_tokens=[latest_token],
                                          constraint_tokens=[constraint_token],
                                          enc_states=enc_states,
                                          dec_init_states=[last_state],
                                          prev_coverage=prev_coverage)

      # score = np.random.random()
      coverage = copy.deepcopy(hyp.coverage)
      coverage[idx][0] = 1
      punctuation_split = hyp.punctuation_split
      if len(coverage[idx]) > 1:
        unfinished_constraint = True
      else:
        unfinished_constraint = False
        punctuation_split = False

      # print('start log_constraint_prob: %.4f'%const_probs[0])
      new_hyp = hyp.extend(token=constraint_token,
                           log_prob=const_probs[0],
                           state=new_states[0],
                           attn_dist=attn_dists[0],
                           p_gen=p_gens[0],
                           s2s_coverage=new_coverage[0],
                           coverage=coverage,
                           constraints=hyp.constraints,
                           payload=None,
                           backpointer=hyp,
                           constraint_index=(idx, 0),
                           unfinished_constraint=unfinished_constraint,
                           punctuation_split=punctuation_split
                           )
      new_constraint_hyps.append(new_hyp)

    return new_constraint_hyps

  def dumb_continue_unfinished_constraint(self, sess, model, batch, hyp, enc_states):
    assert hyp.unfinished_constraint is True, 'hyp must be part of an unfinished constraint'

    # this should come from the model
    # score = np.random.random()

    constraint_row_index = hyp.constraint_index[0]
    # the index of the next token in the constraint
    constraint_tok_index = hyp.constraint_index[1] + 1
    constraint_index = (constraint_row_index, constraint_tok_index)

    continued_constraint_token = hyp.constraints[constraint_index[0]][constraint_index[1]]

    coverage = copy.deepcopy(hyp.coverage)
    coverage[constraint_row_index][constraint_tok_index] = 1

    punctuation_split = hyp.punctuation_split
    if len(hyp.constraints[constraint_row_index]) > constraint_tok_index + 1:
      unfinished_constraint = True
    else:
      unfinished_constraint = False
      punctuation_split = False

    latest_token = hyp.latest_token
    last_state = hyp.state
    prev_coverage = hyp.s2s_coverage

    (next_tokens, next_scores, const_probs, new_states, attn_dists, p_gens, new_coverage) \
      = model.decode_onestep_constraint(sess=sess,
                                        batch=batch,
                                        latest_tokens=[latest_token],
                                        constraint_tokens=[continued_constraint_token],
                                        enc_states=enc_states,
                                        dec_init_states=[last_state],
                                        prev_coverage=prev_coverage)

    # print('continue log_constraint_prob: %.4f' % const_probs[0])
    new_hyp = hyp.extend(token=continued_constraint_token,
                         log_prob=const_probs[0],
                         state=new_states[0],
                         attn_dist=attn_dists[0],
                         p_gen=p_gens[0],
                         s2s_coverage=new_coverage[0],
                         coverage=coverage,
                         constraints=hyp.constraints,
                         payload=None,
                         backpointer=hyp,
                         constraint_index=constraint_index,
                         unfinished_constraint=unfinished_constraint,
                         punctuation_split=punctuation_split
                         )
    return new_hyp


def init_coverage(constraints):
  coverage = []
  for c in constraints:
    coverage.append(np.zeros(len(c), dtype='int16'))
  return coverage


def run_beam_search(sess, model, vocab, ckg, funcWords, black_list, stopWords, batch, consDecoder, counter, trigram,trigram_v2,train_text, dic, source_dict):
  """Performs beam search decoding on the given example.

  Args:
    sess: a tf.Session
    model: a seq2seq model
    vocab: Vocabulary object
    batch: Batch object that is the same example repeated across the batch

  Returns:
    best_hyp: Hypothesis object; the best hypothesis found by beam search.
  """
  # Run the encoder to get the encoder hidden states and decoder initial state
  enc_states, dec_in_state = model.run_encoder(sess, batch)
  # hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING)],
  #                    log_probs=[0.0],
  #                    state=dec_in_state,
  #                    attn_dists=[],
  #                    p_gens=[],
  #                    coverage=np.zeros([batch.enc_batch.shape[1]]) # zero vector of length attention_length
  #                    ) for _ in range(FLAGS.beam_size)]
  constraints_list = batch.constraints_list
  #print('constraints_len：{}'.format(len(constraints_list[0])))
  start_hyps = [ConstraintHypothesis(tokens=[vocab.word2id(data.START_DECODING)],
                                     log_probs=[0.0],
                                     state=dec_in_state,
                                     attn_dists=[],
                                     p_gens=[],
                                     s2s_coverage=np.zeros([batch.enc_batch.shape[1]]),
                                     coverage=init_coverage(constraints),
                                     constraints=constraints,
                                     payload=None,
                                     backpointer=None,
                                     unfinished_constraint=False
                                     ) for constraints in constraints_list]

  grid_height = sum(len(c) for c in constraints_list[0])
  print('grid_height: %d' % grid_height)
  output_grid = consDecoder.search(sess=sess,
                                   model=model,
                                   vocab=vocab,
                                   ckg=ckg,
                                   funcWords=funcWords,
                                   black_list=black_list,
                                   trigram = trigram,
                                   trigram_v2 =trigram_v2,
                                   train_text = train_text,
                                   stopWords=stopWords,
                                   batch=batch,
                                   start_hyp=start_hyps[0],
                                   # constraints=constraints_list[0],
                                   grid_height=grid_height,
                                   enc_states=enc_states,
                                   max_source_len=FLAGS.max_dec_steps,
                                   beam_size=FLAGS.beam_size,
                                   dic = dic,
                                   source_dict = source_dict)
  start = grid_height
  end = FLAGS.max_dec_steps
  tmp_res = []
  final_res = []
  def long_sents_without_juhao(String):
    num_juhao=0
    num_other_fuhao=0
    for w in String:
      if w=='。':
        num_juhao+=1
      if w in ['，','、','！','；','？']:
        num_other_fuhao+=1
    if (num_juhao<2) and (num_other_fuhao>5):
       return True
    return False
  while (start >= 0):
    # for start in range(grid_height - 1, -1, -1):
    res = []
    for i in range(max(start, 1), end + 1):
      beam = output_grid[(i, start)]
      print('beam(%d, %d) length: %d' % (i, start, len(beam)))
      res.extend([hyp for hyp in beam])
    print('height %d contains %d hyps' % (start, len(res)))
    num_notStop = 0
    num_unfishedCons = 0
    num_too_small_tokens = 0
    num_too_small_chars = 0
    num_too_large_chars = 0
    for hyp in res:
      if not (hyp.latest_token == vocab.word2id("。") or hyp.latest_token == vocab.word2id("！")):
        num_notStop += 1
        continue      
      if hyp.unfinished_constraint:
        num_unfishedCons += 1
        continue
      if not len(hyp.tokens) >= FLAGS.min_dec_steps:
        num_too_small_tokens += 1
        continue
      if not len(hyp.plain_text(vocab, batch).strip('，。！？；')) >= 54:
        num_too_small_chars += 1
        continue
      if not len(hyp.plain_text(vocab, batch).strip('，。！？；')) <= 89:
        num_too_large_chars += 1
        continue
      if long_sents_without_juhao(hyp.plain_text(vocab, batch)):
        continue
      final_res.append(hyp)

    print('num_notStop: %d/%d' % (num_notStop, len(res)))
    print('num_unfishedCons: %d/%d' % (num_unfishedCons, len(res)))
    print('num_too_small_tokens: %d/%d' % (num_too_small_tokens, len(res)))
    print('num_too_small_chars: %d/%d' % (num_too_small_chars, len(res)))
    print('num_too_large_chars: %d/%d' % (num_too_large_chars, len(res)))
    start -= 1
    if (len(final_res) > 0):
      print('Failed in Height %d/%d\n' % (start, grid_height))

  final_res = sort_hyps_final(final_res,vocab,batch,trigram_v2,0,0,dic)
  #final_res = sort_hyps(final_res)
  if (len(final_res) > 0):
    print('Instance %d: Success in %d/%d, Full Constraint: %s, FirstChoose, avg_log_p: %.4f\n' % (
      counter, start, grid_height, start == grid_height, final_res[0].avg_log_prob_v2))
    return final_res[0]
  else:
    return None
    print('Failed in Height %d/%d\n' % (start, grid_height))

  sorted_hyps = sort_hyps(tmp_res)
  print('Instance %d: ERROR: len(final_res) == 0, have to return sorted_hyps[0]')
  print('len(sorted_hyps):%d' % len(sorted_hyps))
  if (len(sorted_hyps) > 0):
    print(sorted_hyps[0].plain_text(vocab, batch))
    return sorted_hyps[0]
  else:
    return None
'''
  while (start >= 0 and len(final_res) == 0):
    # for start in range(grid_height - 1, -1, -1):
    res = []
    for i in range(max(start, 1), end + 1):
      beam = output_grid[(i, start)]
      #print('beam(%d, %d) length: %d' % (i, start, len(beam)))
      res.extend([hyp for hyp in beam])
    sorted_hyps = sort_hyps(res)
    #print('height %d contains %d hyps' % (start, len(sorted_hyps)))
    num_notStop = 0
    num_unfishedCons = 0
    num_too_small_tokens = 0
    num_too_small_chars = 0
    num_too_large_chars = 0
    final_res = []
    for hyp in sorted_hyps:
      if not hyp.latest_token == vocab.word2id(data.STOP_DECODING):
        num_notStop += 1
        continue
      if hyp.unfinished_constraint:
        num_unfishedCons += 1
        continue
      if not len(hyp.tokens) >= FLAGS.min_dec_steps:
        num_too_small_tokens += 1
        continue
      if not len(hyp.plain_text(vocab, batch).strip('，。！？；')) >= 49:
        num_too_small_chars += 1
        continue
      if not len(hyp.plain_text(vocab, batch).strip('，。！？；')) <= 89:
        num_too_large_chars += 1
        continue
      final_res.append(hyp)

    #print('num_notStop: %d/%d' % (num_notStop, len(sorted_hyps)))
    #print('num_unfishedCons: %d/%d' % (num_unfishedCons, len(sorted_hyps)))
    #print('num_too_small_tokens: %d/%d' % (num_too_small_tokens, len(sorted_hyps)))
    #print('num_too_small_chars: %d/%d' % (num_too_small_chars, len(sorted_hyps)))
    #print('num_too_large_chars: %d/%d' % (num_too_large_chars, len(sorted_hyps)))

    if (len(final_res) > 0):
      print('Instance %d: Success in %d/%d, Full Constraint: %s, FirstChoose, avg_log_p: %.4f\n' % (
        counter, start, grid_height, start == grid_height, final_res[0].avg_log_prob))
      return final_res[0]
    print('Failed in Height %d/%d\n' % (start, grid_height))
    start -= 1

    # final_res = [hyp for hyp in sorted_hyps if
    #             hyp.latest_token == vocab.word2id(data.STOP_DECODING) and
    #             not hyp.unfinished_constraint and
    #             len(hyp.tokens) >= FLAGS.min_dec_steps and
    #             hyp.char_len(vocab, batch) > 50 and
    #             hyp.char_len(vocab, batch) <= 91]

  start = grid_height
  while (start >= 1 and len(final_res) == 0):
    # for start in range(grid_height - 1, -1, -1):
    res = []
    for i in range(max(start, 1), end + 1):
      beam = output_grid[(i, start)]
      #print('beam(%d, %d) length: %d' % (i, start, len(beam)))
      res.extend([hyp for hyp in beam])
    sorted_hyps = sort_hyps(res)
    #print('height %d contains %d hyps' % (start, len(sorted_hyps)))
    num_unfishedCons = 0
    num_too_small_tokens = 0
    num_too_small_chars = 0
    num_too_large_chars = 0
    final_res = []
    for hyp in sorted_hyps:
      if hyp.unfinished_constraint:
        num_unfishedCons += 1
        continue
      if not len(hyp.tokens) >= FLAGS.min_dec_steps:
        num_too_small_tokens += 1
        continue
      if not len(hyp.plain_text(vocab, batch).strip('，。！？；')) >= 49:
        num_too_small_chars += 1
        continue
      if not len(hyp.plain_text(vocab, batch).strip('，。！？；')) <= 89:
        num_too_large_chars += 1
        continue
      final_res.append(hyp)

    #print('num_notStop: %d/%d' % (num_notStop, len(sorted_hyps)))
    #print('num_unfishedCons: %d/%d' % (num_unfishedCons, len(sorted_hyps)))
    #print('num_too_small_tokens: %d/%d' % (num_too_small_tokens, len(sorted_hyps)))
    #print('num_too_small_chars: %d/%d' % (num_too_small_chars, len(sorted_hyps)))
    #print('num_too_large_chars: %d/%d' % (num_too_large_chars, len(sorted_hyps)))

    if (len(final_res) > 0):
      print('Instance %d: Success in %d/%d, Full Constraint: %s, SecondaryChoose: True, avg_log_p: %.4f\n' % (
        counter, start, grid_height, start == grid_height, final_res[0].avg_log_prob))
      return final_res[0]
    print('Failed in Height %d/%d\n' % (start, grid_height))
    start -= 1

    # final_res = [hyp for hyp in sorted_hyps if
    #             hyp.latest_token == vocab.word2id(data.STOP_DECODING) and
    #             not hyp.unfinished_constraint and
    #             len(hyp.tokens) >= FLAGS.min_dec_steps and
    #             hyp.char_len(vocab, batch) > 50 and
    #             hyp.char_len(vocab, batch) <= 91]

  print('Instance %d: ERROR: len(final_res) == 0, have to return sorted_hyps[0]')
  print('len(sorted_hyps):%d' % len(sorted_hyps))
  if (len(sorted_hyps) > 0):
    print(sorted_hyps[0].plain_text(vocab, batch))
    return sorted_hyps[0]
  else:
    return None
'''
  # expend_rate = 2
  # steps = 0
  # while steps < FLAGS.max_dec_steps and len(results) < FLAGS.beam_size:
  #   latest_tokens = [h.latest_token for h in hyps] # latest token produced by each hypothesis
  #   latest_tokens = [t if t in range(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in latest_tokens] # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
  #   states = [h.state for h in hyps] # list of current decoder states of the hypotheses
  #   prev_coverage = [h.coverage for h in hyps] # list of coverage vectors (or None)
  #
  #   # Run one step of the decoder to get the new info
  #   (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage) = model.decode_onestep(sess=sess,
  #                       batch=batch,
  #                       latest_tokens=latest_tokens,
  #                       enc_states=enc_states,
  #                       dec_init_states=states,
  #                       prev_coverage=prev_coverage)
  #
  #   # Extend each hypothesis and collect them all in all_hyps
  #   all_hyps = []
  #   num_orig_hyps = 1 if steps == 0 else len(hyps) # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
  #   for i in range(num_orig_hyps):
  #     h, new_state, attn_dist, p_gen, new_coverage_i = hyps[i], new_states[i], attn_dists[i], p_gens[i], new_coverage[i]  # take the ith hypothesis and new decoder state info
  #     for j in range(FLAGS.beam_size * expend_rate):  # for each of the top 2*beam_size hyps:
  #       # Extend the ith hypothesis with the jth option
  #       new_hyp = h.extend(token=topk_ids[i, j],
  #                          log_prob=topk_log_probs[i, j],
  #                          state=new_state,
  #                          attn_dist=attn_dist,
  #                          p_gen=p_gen,
  #                          coverage=new_coverage_i)
  #       all_hyps.append(new_hyp)
  #
  #   # Filter and collect any hypotheses that have produced the end token.
  #   newHyps = [] # will contain hypotheses for the next step
  #   for h in sort_hyps(all_hyps): # in order of most likely h
  #     if h.latest_token == vocab.word2id(data.STOP_DECODING): # if stop token is reached...
  #       # If this hypothesis is sufficiently long, put in results. Otherwise discard.
  #       if steps >= FLAGS.min_dec_steps and h.char_len(vocab, batch) > 55 and h.char_len(vocab, batch) <= 90:
  #         results.append(h)
  #     elif h.contain_uniq_bigrams() \
  #         and h.contain_no_redundant_continuous_tokens() \
  #         and h.contain_no_unk(vocab) \
  #         and h.contain_no_badWrod(vocab, batch, black_list) \
  #         and h.contain_uniq_triLetters(vocab, batch) \
  #         and h.not_start_with_specific_token(vocab, batch) \
  #         and h.consistent_with_KB(vocab, batch, ckg): # hasn't reached stop token, so continue to extend this hypothesis
  #       newHyps.append(h)
  #     # else: # hasn't reached stop token, so continue to extend this hypothesis
  #     #   hyps.append(h)
  #     if len(newHyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
  #       # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
  #       break
  #
  #   if(len(newHyps) < FLAGS.beam_size and expend_rate < 4):
  #     expend_rate = expend_rate * 2
  #     continue
  #
  #   expend_rate = 2
  #   steps += 1
  #   hyps = newHyps
  #
  # # At this point, either we've got beam_size results, or we've reached maximum decoder steps
  #
  # if len(results)==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
  #   print('ERROR: len(results)==0')
  #   results = hyps
  #
  # # Sort hypotheses by average log probability
  # hyps_sorted = sort_hyps(results)
  #
  # # Return the hypothesis with highest average log prob
  # return hyps_sorted[0]
def sort_hyps_v5(hyps,vocab,batch,trigram_v2,v1,v2):
   for h in hyps:
     s1 = h.trigram_func_long(vocab,batch,trigram_v2)
     s2 = h.avg_log_prob
     h.s = s1+0.1*s2
   return sorted(hyps, key=lambda h: h.s, reverse=True)

def sort_hyps_v4(hyps,vocab,batch,trigram_v2,v1,v2,dic):
   for h in hyps:
     s1 = h.trigram_func(vocab,batch,trigram_v2,dic)
     s2 = h.avg_log_prob
     h.s = s1+0.1*s2
   return sorted(hyps, key=lambda h: h.s, reverse=True)
def sort_hyps_final(hyps,vocab,batch,trigram_v2,v1,v2,dic):
   for h in hyps:
     s1 = h.trigram_func_final(vocab,batch,trigram_v2,dic)
     s2 = h.avg_log_prob
     h.s = s1+0.1*s2
   return sorted(hyps, key=lambda h: h.s, reverse=True)
def sort_hyps_v3(hyps,v1,v2):
  return sorted(hyps, key=lambda h: v1*h.avg_log_prob_v2+v2*h.avg_log_prob, reverse=True)
def sort_hyps_v2(hyps):
  return sorted(hyps, key=lambda h: h.avg_log_prob_v2, reverse=True) 
def sort_hyps(hyps):
  """Return a list of Hypothesis objects, sorted by descending average log probability"""
  return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
