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
               payload=None, backpointer=None,
               unfinished_constraint=False, punctuation_split=True):

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
    self.s2s_coverage = s2s_coverage
    self.backpointer = backpointer
    self.payload = payload
    self.unfinished_constraint = unfinished_constraint
    self.punctuation_split = punctuation_split
    self.s = 0
    self.word_dict={}
    self.res=[]
#     pdb.set_trace()
    
  def __str__(self):
    return u'token: {}, sequence: {}, score: {}'.format(
      self.token, self.sequence, self.score)

  def __getitem__(self, key):
    return getattr(self, key)

  def extend(self, token, tokenStr, log_prob, state, attn_dist, p_gen, s2s_coverage,
             payload=None, backpointer=None,
             unfinished_constraint=False, punctuation_split=True):
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
                                payload=payload,
                                backpointer=backpointer,
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

  def should_not_be_pruned(self, vocab, batch, train_text, isEval):
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

  def add(self, hyp,vocab,batch,v1,v2, res):
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

  def search(self, sess, model, vocab, batch,
             start_hyp, enc_states, enc_aspects, selector_probs, max_source_len, beam_size,
             train_text, eval):
    search_grid = OrderedDict()
    result_ = []
    start_beam = self.beam_implementation(size=beam_size)
    v1 = model._hps.arg1
    v2 = model._hps.arg2
    start_beam.vocab=vocab
    start_beam.batch=batch
    trigram_=train_text.getGram()
    ori_text = trigram_ 
    for i in range(beam_size):
      start_beam.add(start_hyp,vocab,batch,v1,v2, result_)
    search_grid[(0, 0)] = start_beam
    for i in range(1, max_source_len + 1):
      j_start = 0
      j_end = min(i, 0) + 1
      beams_in_i = j_end - j_start
      for j in range(j_start, j_end):
        new_beam = self.beam_implementation(size=beam_size)
        if (i - 1, j) in search_grid:
          generation_hyps = self.get_generation_hyps(
            sess=sess, model=model, vocab=vocab, batch=batch, steps=i-1, beam=search_grid[(i - 1, j)],
            enc_states=enc_states, enc_aspects=enc_aspects, selector_probs=selector_probs, train_text=ori_text, eval=eval)
          sort_tmp = sort_hyps(generation_hyps)
          #if i==50:
          #  for ww in sort_tmp:
          #      print(' '.join([ str(w) for w in ww.tokens]))
          for hyp in sort_tmp:
            new_beam.add(hyp,vocab,batch,v1,v2, result_)
          
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
            if hyp.should_not_be_pruned(vocab=vocab, batch=batch, train_text=ori_text,isEval=eval):
              new_beam.add(hyp,vocab,batch,v1,v2)
          for hyp in continued_constraint_hyps:
            if hyp.should_not_be_pruned(vocab=vocab, batch=batch, train_text=ori_text,isEval=eval):
              new_beam.add(hyp,vocab,batch,v1,v2)

        if(len(new_beam) > 0 and len(new_beam) != FLAGS.beam_size):
#           pdb.set_trace()
          tmp_hyp = new_beam[0]
          left = FLAGS.beam_size - len(new_beam)
          for a in range(left):
            new_beam.add(tmp_hyp,vocab,batch,v1,v2,result_)
        
        search_grid[(i, j)] = new_beam
#         pdb.set_trace()
    return result_ 
  
  def get_generation_hyps(self, sess, model, vocab, batch, steps, beam, enc_states, enc_aspects, selector_probs, train_text, eval):
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
    (next_tokens, next_scores, new_states, attn_dists, p_gens, new_coverage) \
      = model.decode_onestep_constraint(sess=sess,
                                        batch=batch,
                                        latest_tokens=latest_tokens,
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
                             payload=None,
                             backpointer=hyp,
                             constraint_index=None,
                             unfinished_constraint=False,
                             punctuation_split=punctuation_split
                             )
        if new_hyp.should_not_be_pruned(vocab=vocab, batch=batch, train_text=train_text, isEval=eval):
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
      if new_hyp.should_not_be_pruned(vocab=vocab, batch=batch, train_text=train_text,isEval=eval):
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


def run_beam_search(sess, model, vocab, batch, consDecoder, train_text, eval):
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
                                     payload=None,
                                     backpointer=None,
                                     unfinished_constraint=False
                                     )]

  #print('grid_height: %d' % grid_height)
  output_grid = consDecoder.search(sess=sess,
                                   model=model,
                                   vocab=vocab,
                                   train_text = train_text,
                                   batch=batch,
                                   start_hyp=start_hyps[0],
                                   enc_states=enc_states,
                                   enc_aspects=enc_aspects,
                                   selector_probs=selector_probs,
                                   max_source_len=FLAGS.max_dec_steps,
                                   beam_size=FLAGS.beam_size,
                                   eval = eval)
#   pdb.set_trace()
  output_grid = sort_hyps(output_grid)
  start = 0
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
