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

import threading
from multiprocessing import Process
import jieba
import copy
import tensorflow as tf
import numpy as np
import data
from collections import defaultdict, OrderedDict
import re
import time
from flashtext import KeywordProcessor
import util
import pdb
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

  def __init__(self, tokens, tokenStrs, pre_text, pre_bigram_set, pre_triLetter_set, log_probs, state, attn_dists, p_gens, s2s_coverage,
               coverage, constraints, payload=None, backpointer=None,
               constraint_index=None, unfinished_constraint=False, punctuation_split=True):

    assert len(coverage) == len(constraints), 'constraints and coverage length must match'
    assert all(len(cov) == len(cons) for cov, cons in zip(coverage, constraints)), \
      'each coverage and constraint vector must match'

    self.tokens = tokens
    self.tokenStrs = tokenStrs
    


    
#     self.pre_bigram_set = set()
#     for i in range(len(self.tokens) - 2):
#       bigram = self.tokenStrs[i] + "\t" + self.tokenStrs[i + 1]  # 判断 bigram
#       self.pre_bigram_set.add(bigram)

    self.pre_bigram_set = set(pre_bigram_set)
    self.cur_bigram_set = set(pre_bigram_set)
    self.last_bigram = self.tokenStrs[len(tokens) - 2] + '\t' + self.tokenStrs[len(tokens) - 1] if len(tokens) > 2 else ''
    if self.last_bigram != '':
      self.cur_bigram_set.add(self.last_bigram)
    
    self.pre_text = pre_text
#     pattern = re.compile(r'[^\u4e00-\u9fa5，。？；！、]+$')
    if len(tokens) > 1:
      if (tokenStrs[-1] == data.STOP_DECODING):
        self.plain_text = pre_text
      else:
#         if len(tokens) > 2 and re.match(pattern, tokenStrs[-2] + ' ' + tokenStrs[-1]):
#           self.plain_text = (pre_text + ' ' + self.tokenStrs[-1]).strip()
#         else:
#           self.plain_text = (pre_text + self.tokenStrs[-1]).strip()
        self.plain_text = (pre_text + self.tokenStrs[-1]).strip()
    else:
      self.plain_text = ''
#     self.plain_text = pre_text + self.tokenStrs[-1] if len(tokens) > 1 else ''
#     self.plain_text = ''.join(tokenStrs[1:]).replace(' ', '').replace('\t', '').replace(data.STOP_DECODING, '')
#     self.pre_text = ''.join(tokenStrs[1:-1]).replace(' ', '').replace('\t', '').replace(data.STOP_DECODING, '')
    
#     self.pre_triLetter_set = set()
    self.pre_triLetter_set = set(pre_triLetter_set)
    self.cur_triLetter_set = set(pre_triLetter_set)
    self.last_triLetters = set()
    if len(self.plain_text) > 2 and self.tokenStrs[-1] != data.STOP_DECODING:
      index = len(self.pre_text) 
      for i in range(len(self.tokenStrs[-1])):
        if index + i - 2 >= 0:
          triLetter = self.plain_text[index + i - 2] + "\t" + self.plain_text[index + i -1] + "\t" + self.plain_text[index + i]
          self.last_triLetters.add(triLetter)
    
    if len(self.last_triLetters) != 0:
      for triLetter in self.last_triLetters:
        self.cur_triLetter_set.add(triLetter)      
#     for i in range(len(self.plain_text) - 4, -1, -1):  # 判断triletter
#       triLetter = str(self.plain_text[i]) + "\t" + str(self.plain_text[i + 1]) + "\t" + str(self.plain_text[i + 2])
#       self.pre_triLetter_set.add(triLetter)
      
        
#     self.plain_text[len(self.plain_text) - 3] + '\t' + self.plain_text[len(self.plain_text) - 2] + '\t' + self.plain_text[len(self.plain_text) - 1] if len(self.plain_text) > 2 else ''
    
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
    self.res=[]
#     pdb.set_trace()
    
  def __str__(self):
    return u'token: {}, sequence: {}, score: {}, coverage: {}, constraints: {},'.format(
      self.token, self.sequence, self.score, self.coverage, self.constraints)

  def __getitem__(self, key):
    return getattr(self, key)

  def extend(self, token, tokenStr, log_prob, state, attn_dist, p_gen, s2s_coverage,
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
                                tokenStrs=self.tokenStrs + [tokenStr],
                                pre_text=self.plain_text,
                                pre_bigram_set = self.cur_bigram_set,
                                pre_triLetter_set = self.cur_triLetter_set,
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

  def trigram_func(self,vocab,batch,trigram_v2,dic):
    sku_kb = batch.sku_kbs[0]
    cat3 = sku_kb['CategoryL3'.lower()][0].replace('/', '_')
    trigram1 = trigram_v2.getGram()
    pred_word = self.tokenStrs
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
  
  def digit_consistent(self, vocab, batch, inputValues, inputLongValues, inputValueUnits):
    t1 = time.time()
    pred_word = self.plain_text
#     valuePattern = re.compile(r'[^\u4e00-\u9fa5，。？；！、 ]+')
    longValuePattern = re.compile(r'[^\u4e00-\u9fa5，。？；！、]+')
    valueUnitPattern = re.compile(r'[^\u4e00-\u9fa5，。？；！、 ]+[\u4e00-\u9fa5，。？；！、]')

#     values = [_.strip() for _ in re.findall(valuePattern, pred_word) if _.strip() != '']
    longValues = [_.strip() for _ in re.findall(longValuePattern, pred_word) if _.strip() != '']
    valueUnits = [_.strip() for _ in re.findall(valueUnitPattern, pred_word) if len(_.strip()) > 1]

#     if len(values) == 0 and len(longValues) == 0 and len(valueUnits) == 0:
#     if len(values) == 0 and len(valueUnits) == 0:
    if len(longValues) == 0 and len(valueUnits) == 0:
      return True
#     inputInfo = caption.lower()    
#     inputNums = [_.strip() for _ in re.findall(longValuePattern, inputInfo) if len(_.strip()) > 1]
#     if any([_ not in inputValues for _ in values]):
# #       pdb.set_trace()
#       return False
    if any([_ not in inputLongValues for _ in longValues]):
#       pdb.set_trace()
      return False
    if any([_ not in inputValueUnits for _ in valueUnits]):
#       pdb.set_trace()
      return False
    t2 = time.time()
    return True
  
  def new_rule(self, vocab, batch, caption, black_list, stopWords, source_dict, confictValues, confictValueUnits, inputValues, inputLongValues, inputValueUnits, bigramlm):
    pred_word = self.tokenStrs
    lastWord = pred_word[-1]
    previousWords = pred_word[:-1]
    pred_text = self.plain_text
    if vocab.word2id(data.UNKNOWN_TOKEN) in self.tokens:    # 判断UNK
#       pdb.set_trace()
      return False
    if pred_text.startswith('这款'):      # 判断开头
#       pdb.set_trace()
      return False
    if lastWord not in stopWords and lastWord in previousWords:    # 判断unigram
#       pdb.set_trace()
      return False
    if len(pred_word) > 1 and pred_word[-1] == pred_word[-2]: ##叠词
#       pdb.set_trace()
      return False
#     pdb.set_trace()
    if pred_word[-2] != data.START_DECODING and pred_word[-1] != data.STOP_DECODING and bigramlm.scoring(pred_word) <= 0.001:
      return False      
    longPreviousWords = [_ for _ in previousWords if len(_) >= 2]
    if lastWord not in stopWords and len(lastWord) >= 2 and any([lastWord in _ or _ in lastWord for _ in longPreviousWords]):
#       pdb.set_trace()
      return False
    if any([confictValueUnit in pred_text for confictValueUnit in confictValueUnits]):
#       pdb.set_trace()
      return False
    if confictValues.conflic(caption):
      return False
    if confictValues.conflic(pred_text):
      return False
    if self.last_bigram in self.pre_bigram_set:
#       pdb.set_trace()
      return False
    if any([_ in self.pre_triLetter_set for _ in self.last_triLetters]):
#       pdb.set_trace()
      return False
    if not self.digit_consistent(vocab, batch, inputValues, inputLongValues, inputValueUnits): #last_token是否有包含数字+字母
#       pdb.set_trace()
      return False

#     bigram_set = set()
#     for i in range(len(self.tokens) - 1):
#       bigram = str(self.tokens[i]) + "\t" + str(self.tokens[i + 1])  # 判断 bigram
#       if (bigram in bigram_set):
#         return False
#       bigram_set.add(bigram)

#     triLetter_set = set()
#     for i in range(len(pred_text) - 3, -1, -1):  # 判断triletter
#       triLetter = str(pred_text[i]) + "\t" + str(pred_text[i + 1]) + "\t" + str(pred_text[i + 2])
#       if (triLetter in triLetter_set):
#         return False
#       triLetter_set.add(triLetter)

#     if black_list.extract_keywords(pred_text):  # 判断black
#      return False

    source_find = source_dict.extract_keywords(pred_text)  # 判断origin
    if source_find:
      for find in source_find:
        if not find.lower() in caption.lower():
          return False
        
    return True
  
  def have_dic(self, vocab, batch, train_text, word_dict):
    pred_word = self.tokenStrs
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
      tmp_sen = pred_word[end_l:end+1]
      tmp_str = ''.join(pred_word[end-1:end+1])
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
          return False
        str_tmp = '$'.join(dic_list)
        word_tmp = jieba.lcut(str_tmp)
        word_dict[cat3+'_'+tmp_str]=word_tmp
      time2 = time.time()
      
      txt_word_dic = ' '.join(word_tmp)
      if txt_word_dic.find(pred_word[-1])<0:
         return False

      return True

  def long_sents_without_punc(self, vocab, batch, max_len=15, max_num_word=10, min_len=3, min_num_word=2):
    pred = self.plain_text
    pred_word = self.tokenStrs
    num_words = len(pred_word)
    if num_words<3:
      return True
    tmp = re.split(r"，|。|、|；|！", pred)
    for t in tmp:
      if len(t)>max_len:
#       pdb.set_trace()
        return False

#     punc_list=["，", "。", "！", "、", "；"]
#     if pred_word[-1] in punc_list:
#       tmp = re.split(r"，|。|、|；|！", pred)
#       for t in tmp:
#         if len(t)>max_len:
# #           pdb.set_trace()
#           return False
    return True
  
  def should_not_be_pruned(self, vocab, batch, ckg, funcWords, black_list, stopWords, trigram, train_text, dic, source_dict, bigramlm, confictValues, isEval):
    if isEval:
      return True
    # 提前取出caption
#     sku_kb = batch.sku_kbs[0]
#     kb_str=''
#     for d in sku_kb:
#       kb_str+=d
#       kb_str+=''.join(sku_kb[d])
#     ocr_shortSeqs = ''.join(batch.ocr_shortSeqs_list[0])
# #     caption = ''.join(batch.caption_tokens_list[0])
#     caption = util.join_tokens(batch.caption_tokens_list[0])
#     caption+=kb_str
    if len(self.tokens) == 1:
      return True
    
    caption = batch.inputInfo[0]
    confictValueUnits = batch.confictValueUnitsList[0]
    inputValues = batch.inputValuesList[0]   
    inputLongValues = batch.inputLongValuesList[0]   
    inputValueUnits = batch.inputValueUnitsList[0]   

   # if self.new_rule(vocab=vocab, batch=batch, caption=caption, black_list=black_list, stopWords=stopWords, source_dict=source_dict, confictValues=confictValues, confictValueUnits=confictValueUnits, inputValues=inputValues, inputLongValues=inputLongValues, inputValueUnits=inputValueUnits, bigramlm=bigramlm) \
    #    and self.long_sents_without_punc(vocab, batch, max_len=15, max_num_word=50, min_len=4, min_num_word=2):
    #       return True
    return True
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

  def __init__(self, size):
    # note: here we assume bigger scores are better
    # self.hypotheses = SortedListWithKey(key=lambda x: -x['score'])
    self.hypotheses = []
    self.size = size

  def add(self, hyp,vocab,batch,trigram_v2,v1,v2,dic, res):
    if len(res)==FLAGS.beam_size or len(self.hypotheses)==FLAGS.beam_size:
      return
    if hyp.latest_token == vocab.word2id(data.STOP_DECODING):
#       if len(hyp.plain_text)>54:
      res.append(hyp)
    else:
      self.hypotheses.append(hyp)
      self.hypotheses = sort_hyps(self.hypotheses)
       #self.hypotheses = sort_hyps_v4(self.hypotheses,vocab,batch,trigram_v2,v1,v2,dic)

  def __len__(self):
    return len(self.hypotheses)

  def __iter__(self):
    for hyp in self.hypotheses:
      yield hyp
  def __getitem__(self, idx):
    return self.hypotheses[idx]

class ConstrainedDecoder(object):
  def __init__(self, beam_implementation=AbstractBeam):
    self.beam_implementation = beam_implementation

  def search(self, sess, model, vocab, ckg, funcWords, black_list, stopWords, batch,
             start_hyp, grid_height, enc_states, enc_aspects, selector_probs, max_source_len, beam_size, trigram,
             trigram_v2,train_text, dic, source_dict, bigramlm, confictValues, eval):
    search_grid = OrderedDict()
    result_ = []
    start_beam = self.beam_implementation(size=beam_size)
    v1 = model._hps.arg1
    v2 = model._hps.arg2
    start_beam.vocab=vocab
    start_beam.batch=batch
    start_beam.trigram_v2=trigram_v2
    trigram_=train_text.getGram()
    ori_text = trigram_ 
    for i in range(beam_size):
      start_beam.add(start_hyp,vocab,batch,trigram_v2,v1,v2,dic, result_)
    search_grid[(0, 0)] = start_beam
    for i in range(1, max_source_len + 1):
      j_start = 0
      j_end = min(i, grid_height) + 1
      beams_in_i = j_end - j_start
      for j in range(j_start, j_end):
        new_beam = self.beam_implementation(size=beam_size)
        if (i - 1, j) in search_grid:
          generation_hyps = self.get_generation_hyps(
            sess=sess, model=model, vocab=vocab, batch=batch, steps=i-1, beam=search_grid[(i - 1, j)],
            enc_states=enc_states, enc_aspects=enc_aspects, selector_probs=selector_probs, ckg=ckg, funcWords=funcWords, black_list=black_list, stopWords=stopWords, trigram=trigram, train_text=ori_text, dic=dic, source_dict= source_dict, bigramlm=bigramlm, confictValues=confictValues, eval=eval)
          sort_tmp = sort_hyps(generation_hyps)
          #if i==50:
          #  for ww in sort_tmp:
          #      print(' '.join([ str(w) for w in ww.tokens]))
          for hyp in sort_tmp:
            new_beam.add(hyp,vocab,batch,trigram_v2,v1,v2,dic, result_)
          
          #if i==8:
          #  for ww in new_beam:
          #      print(' '.join([ str(w) for w in ww.tokens]))
#           pdb.set_trace()
        if (i - 1, j - 1) in search_grid:
          new_constraint_hyps = self.get_new_constraint_hyps(
            sess=sess, model=model, vocab=vocab, batch=batch, beam=search_grid[(i - 1, j - 1)], enc_states=enc_states)
          continued_constraint_hyps = self.get_continued_constraint_hyps(
            sess=sess, model=model, vocab=vocab, batch=batch, beam=search_grid[(i - 1, j - 1)], enc_states=enc_states)
          for hyp in new_constraint_hyps:
            if hyp.should_not_be_pruned(vocab=vocab, batch=batch, ckg=ckg, funcWords=funcWords, black_list=black_list, stopWords=stopWords, trigram=trigram, train_text=ori_text,dic=dic, source_dict=source_dict, isEval=eval):
              new_beam.add(hyp,vocab,batch,trigram_v2,v1,v2,dic)
          for hyp in continued_constraint_hyps:
            if hyp.should_not_be_pruned(vocab=vocab, batch=batch, ckg=ckg, funcWords=funcWords, black_list=black_list, stopWords=stopWords, trigram=trigram, train_text=ori_text,dic=dic, source_dict=source_dict, isEval=eval):
              new_beam.add(hyp,vocab,batch,trigram_v2,v1,v2,dic)

        if(len(new_beam) > 0 and len(new_beam) != FLAGS.beam_size):
#           pdb.set_trace()
          tmp_hyp = new_beam[0]
          left = FLAGS.beam_size - len(new_beam)
          for a in range(left):
            new_beam.add(tmp_hyp,vocab,batch,trigram_v2,v1,v2,dic,result_)
        
        search_grid[(i, j)] = new_beam
#         pdb.set_trace()
    return result_ 
  
  def get_generation_hyps_back(self, sess, model, vocab, batch, beam, enc_states, ckg, funcWords, black_list, stopWords, trigram, train_text, dic, source_dict, l1, l2, eval):
    """return all hyps which are continuations of the hyps on this beam

    hyp_generation_func maps `(hyp) --> continuations`

    the coverage vector of the parent hyp is not modified in each child
    """

    continuations = (
    self.dumb_generate(sess=sess, model=model, vocab=vocab, batch=batch, hyp=hyp, enc_states=enc_states,
                       ckg=ckg, funcWords=funcWords, black_list=black_list, stopWords=stopWords, n_best=FLAGS.beam_size, trigram=trigram, train_text=train_text,dic=dic, source_dict=source_dict, l1=l1, l2=l2, eval=eval)
    for hyp in
    beam if
    not hyp.unfinished_constraint and hyp.latest_token != vocab.word2id(data.STOP_DECODING))
    # flatten
    return (new_hyp for hyp_list in continuations for new_hyp in hyp_list)
  
  def get_generation_hyps(self, sess, model, vocab, batch, steps, beam, enc_states, enc_aspects, selector_probs, ckg, funcWords, black_list, stopWords, trigram, train_text, dic, source_dict, bigramlm, confictValues, eval):
    """return all hyps which are continuations of the hyps on this beam

    hyp_generation_func maps `(hyp) --> continuations`

    the coverage vector of the parent hyp is not modified in each child
    """
    if(len(beam) != FLAGS.beam_size):
      return ()
    latest_tokens = [hyp.latest_token if hyp.latest_token<vocab.size() else vocab.word2id(data.UNKNOWN_TOKEN) for hyp in beam]
    states = [hyp.state for hyp in beam]
    prev_coverage = [hyp.s2s_coverage for hyp in beam]
    constraint_tokens = [vocab.word2id(data.PAD_TOKEN) for hyp in beam]
#     print('Step:%d--%d'%(steps, len(beam)))
#     pdb.set_trace()
    (next_tokens, next_scores, const_probs, new_states, attn_dists, p_gens, new_coverage) \
      = model.decode_onestep_constraint(sess=sess,
                                        batch=batch,
                                        latest_tokens=latest_tokens,
                                        constraint_tokens=constraint_tokens,
                                        enc_states=enc_states,
                                        enc_aspects=enc_aspects,
                                        dec_init_states=states,
                                        selector_probs=selector_probs,
                                        prev_coverage=prev_coverage)

    continuations = []
    num_orig_hyps = 1 if steps == 0 else len(beam) # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
    b = 0
    for hyp in beam:
      new_hyps = []
      i = 0
      valid_cnt = 0
#     while (valid_cnt < n_best*2 and i < 100):
      while (valid_cnt < FLAGS.beam_size*2 and i < FLAGS.beam_size*2):
        tok = data.outputids2words([next_tokens[b, i]], vocab, (batch.art_oovs[b] if FLAGS.pointer_gen else None))[0]
        punctuation_split = True if tok in ['，', '。', '；', '！', '、'] else hyp.punctuation_split
        new_hyp = hyp.extend(token=next_tokens[b, i],
                             tokenStr=tok,
                             log_prob=next_scores[b, i],
                             state=new_states[b],
                             attn_dist=attn_dists[b],
                             p_gen=p_gens[b],
                             s2s_coverage=new_coverage[b],
                             coverage=copy.deepcopy(hyp.coverage),
                             constraints=hyp.constraints,
                             payload=None,
                             backpointer=hyp,
                             constraint_index=None,
                             unfinished_constraint=False,
                             punctuation_split=punctuation_split
                             )
        if new_hyp.should_not_be_pruned(vocab=vocab, batch=batch, ckg=ckg, funcWords=funcWords, black_list=black_list, stopWords=stopWords, trigram=trigram,train_text=train_text,dic=dic,source_dict=source_dict, bigramlm=bigramlm, confictValues=confictValues, isEval=eval):
          new_hyps.append(new_hyp)
          valid_cnt += 1
        i += 1
      continuations.append(new_hyps)
      b += 1
      if(steps == 0):
        break
    
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

  def dumb_generate(self, sess, model, vocab, batch, hyp, enc_states, ckg, funcWords, black_list, stopWords, n_best, trigram,train_text,dic, source_dict, l1, l2, eval):
    # make k_best random hyp objects
    latest_token = hyp.latest_token
    latest_token = latest_token if latest_token<vocab.size() else 0
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
#     while (valid_cnt < n_best*2 and i < 100):
    while (valid_cnt < n_best*2 and i < n_best*2):
      tok = data.outputids2words([next_tokens[0, i]], vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))[0]
      punctuation_split = True if tok in ['，', '。', '；', '！', '、'] else hyp.punctuation_split
      new_hyp = hyp.extend(token=next_tokens[0, i],
                           tokenStr=tok,
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
      if new_hyp.should_not_be_pruned(vocab=vocab, batch=batch, ckg=ckg, funcWords=funcWords, black_list=black_list, stopWords=stopWords, trigram=trigram,train_text=train_text,dic=dic,source_dict=source_dict, l1=l1, l2=l2, isEval=eval):
        new_hyps.append(new_hyp)
        valid_cnt += 1
      i += 1

    return new_hyps

  def dumb_generate_from_constraints(self, sess, model, batch, hyp, enc_states, enc_aspects):
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
                                          enc_aspects=enc_aspects,
                                          dec_init_states=[last_state],
                                          prev_coverage=prev_coverage)

      coverage = copy.deepcopy(hyp.coverage)
      coverage[idx][0] = 1
      punctuation_split = hyp.punctuation_split
      if len(coverage[idx]) > 1:
        unfinished_constraint = True
      else:
        unfinished_constraint = False
        punctuation_split = False

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


def run_beam_search(sess, model, vocab, ckg, funcWords, black_list, stopWords, batch, consDecoder, counter, trigram,trigram_v2,train_text, dic, source_dict, bigramlm, confictValues, eval):
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
  enc_states, enc_aspects, dec_in_state, selector_probs = model.run_encoder(sess, batch)
  
  constraints_list = batch.constraints_list
  start_hyps = [ConstraintHypothesis(tokens=[vocab.word2id(data.START_DECODING)],
                                     tokenStrs=[data.START_DECODING],
                                     pre_text='',
                                     pre_bigram_set = set(),
                                     pre_triLetter_set = set(),
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
  #print('grid_height: %d' % grid_height)
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
                                   enc_aspects=enc_aspects,
                                   selector_probs=selector_probs,
                                   max_source_len=FLAGS.max_dec_steps,
                                   beam_size=FLAGS.beam_size,
                                   dic = dic,
                                   source_dict = source_dict,
                                   bigramlm = bigramlm,
                                   confictValues=confictValues,
                                   eval = eval)
#   pdb.set_trace()
  output_grid = sort_hyps(output_grid)
  start = grid_height
  end = FLAGS.max_dec_steps
  tmp_res = []
  final_res = []
  while (start >= 0):
    for hyp in output_grid:
      final_res.append(hyp)
      if len(final_res)== FLAGS.beam_size:
        break
    start -= 1
  final_res = sort_hyps(final_res)
#   if (len(final_res) > 0):
#     return final_res[0]
#   else:
#     return None
  if (len(final_res) > 0):
    return final_res[0], selector_probs
  else:
    return None, selector_probs
    #if not len(hyp.plain_text(vocab, batch).strip('，。！？；')) <= 89 :
      #  num_too_large_chars += 1
      #  continue




def sort_hyps_v4(hyps,vocab,batch,trigram_v2,v1,v2,dic):
  for h in hyps:
    s1 = h.trigram_func(vocab,batch,trigram_v2,dic)
    s2 = h.avg_log_prob
    h.s = s1+0.1*s2
  return sorted(hyps, key=lambda h: h.s, reverse=True)

def sort_hyps(hyps):
  """Return a list of Hypothesis objects, sorted by descending average log probability"""
  return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
