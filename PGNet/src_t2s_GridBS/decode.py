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

"""This file contains code to run beam search decoding, including running ROUGE evaluation and producing JSON datafiles for the in-browser attention visualizer, which can be found here https://github.com/abisee/attn_vis"""

import os
import time
import tensorflow as tf
import beam_search_force_decode
import beam_search
import data
import json
# import pyrouge
import util
import logging
import numpy as np
import sys
import pickle

FLAGS = tf.app.flags.FLAGS

SECS_UNTIL_NEW_CKPT = sys.maxsize  # max number of seconds before loading new checkpoint


class BeamSearchDecoder(object):
  """Beam search decoder."""

  def __init__(self, model, batcher, vocab,  constraintDecoder, train_text, model_path='train'):
    """Initialize decoder.

    Args:
      model: a Seq2SeqAttentionModel object.
      batcher: a Batcher object.
      vocab: Vocabulary object
    """
    self._model = model
    self._model.build_graph()
    self._batcher = batcher
    self._vocab = vocab
    self.constraintDecoder = constraintDecoder
    self._saver = tf.train.Saver()  # we use this to load checkpoints for decoding
    self._sess = tf.Session(config=util.get_config())
    self._dic = {}
    # Load an initial checkpoint to use for decoding
    ckpt_path = util.load_ckpt(self._saver, self._sess, model_path)

    if FLAGS.single_pass:
      # Make a descriptive decode directory name
      ckpt_name = "ckpt-" + ckpt_path.split('-')[-1]  # this is something of the form "ckpt-123456"
      self.step = ckpt_path.split('-')[-1]
      self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name)+str(self._model._hps.arg1)+'_'+str(self._model._hps.arg2))
      if os.path.exists(self._decode_dir):
        raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)

    else:  # Generic decode dir name
      self._decode_dir = os.path.join(FLAGS.log_root, "decode")

    # Make the decode dir if necessary
    if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)

    if FLAGS.single_pass:
      # Make the dirs to contain output written in the correct format for pyrouge
      self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
      if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
      self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
      if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)

  def decode(self, decode_id_path, eval=False):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t_start = time.time()
    t0 = t_start
    counter = 0
    flags = [_.strip() for _ in open(decode_id_path, 'r', encoding='utf-8').readlines()]
    #print('flags length {0}'.format(len(flags)))
    print('start'+str(time.time()))
    num=0
    while True:
      batch = self._batcher.next_batch()
      if batch is None:  # finished decoding dataset in single_pass mode
        assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
        tf.logging.info("Decoder has finished reading dataset for single_pass.")
        tf.logging.info("Output has been saved in %s and %s. Now starting ROUGE eval...", self._rouge_ref_dir,
                        self._rouge_dec_dir)
        """results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)"""
        break

      flag = flags[counter]
      original_article = batch.original_articles[0]  # string
      original_abstract = batch.original_abstracts[0]  # string
      original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings

      article_withunks = data.show_art_oovs(original_article, self._vocab)  # string
      abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab,
                                             (batch.art_oovs[0] if FLAGS.pointer_gen else None))  # string

      # Run beam search to get best Hypothesis
      best_hyp = None
      best_hyp = beam_search_force_decode.ConstraintHypothesis(tokens=[self._vocab.word2id(data.START_DECODING)],
                                                               tokenStrs=[data.START_DECODING],
                                                               pre_text='',
                                                               pre_bigram_set = set(),
                                                               pre_triLetter_set = set(),
                                                               log_probs=[0.0],
                                                               state=None,
                                                               attn_dists=[],
                                                               p_gens=[],
                                                               s2s_coverage=np.zeros([batch.enc_batch.shape[1]]),
                                                               payload=None,
                                                               backpointer=None,
                                                               unfinished_constraint=False
                                                              )
      if flag == '1':
        num += 1
        if FLAGS.force_decoding:
          t1 = time.time()
          best_hyp, selector_probs = beam_search_force_decode.run_beam_search(
            self._sess, self._model, self._vocab, batch,
            self.constraintDecoder, self.train_text, eval)
#           import pdb
#           pdb.set_trace()
          t2 = time.time()
        else:
          best_hyp = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch)

      # Extract the output ids from the hypothesis and convert back to words
        decoded_file_selector_prob = os.path.join(self._rouge_dec_dir, "selector_prob.txt")
#         decoded_file_art_mask = os.path.join(self._rouge_dec_dir, "art_mask.txt")

      output_ids = []
      if best_hyp == None:
        self.write_for_rouge(original_abstract_sents, ['NULL'],['0'],
                                     counter)
        counter += 1
        self.write_for_rouge(original_abstract_sents, ['\n'],['0'],
                                             counter)
      if best_hyp != None:
        #for j in best_hyp:
        j = best_hyp
        output_ids = [int(t) for t in j.tokens[1:]]
        decoded_words = data.outputids2words(output_ids, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))
        decoded_words = data.detokenize(decoded_words)

      # Remove the [STOP] token from decoded_words, if necessary
        try:
          fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
          decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
          decoded_words = decoded_words
        if (len(decoded_words) > 0 and decoded_words[-1] in ["，", ",", ";"]):
          del decoded_words[-1]
        if len(decoded_words) > 0 and decoded_words[-1] not in ["。", "！", "？", ".", "!", "?"]:
          decoded_words.append('。')
      # if len(decoded_words) > 0 and decoded_output[-1] not in ["。","！","？",".","!","?"]:
      #   decoded_output += "。"
        decoded_output = ' '.join(decoded_words)  # single string
#         if len(decoded_words) < 10:
#           decoded_words.append('NULL')
#           decoded_output = "NULL"

        if FLAGS.single_pass:
          self.write_for_rouge(original_abstract_sents, decoded_words,j.log_probs,
                           counter)  # write ref summary and decoded summary to file, to eval with pyrouge later
        else:
          print_results(article_withunks, abstract_withunks, decoded_output)  # log output to screen
          self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists,
                             best_hyp.p_gens)  # write info to .json file for visualization tool
        self.write_for_rouge(original_abstract_sents, ['\n'],[0],
                                     counter)
        counter += 1 
        # Check if SECS_UNTIL_NEW_CKPT has elapsed; if so return so we can load a new checkpoint
        t1 = time.time()
        if t1 - t0 > SECS_UNTIL_NEW_CKPT:
          tf.logging.info('We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint',
                        t1 - t0)
          _ = util.load_ckpt(self._saver, self._sess)
          t0 = time.time()
      #if counter == 10:
      #  break
    t_end = time.time()
    print('end_time'+str(t_end))
    print('1/QPS: %.3f' %((t_end - t_start)/num))

  def write_for_rouge(self, reference_sents, decoded_words,log_probs,ex_index):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

    Args:
      reference_sents: list of strings
      decoded_words: list of strings
      ex_index: int, the index with which to label the files
    """
    # First, divide decoded output into sentences
    decoded_sents = []
    while len(decoded_words) > 0:
      try:
        fst_period_idx = decoded_words.index("。")
      except ValueError:  # there is text remaining that doesn't end in "。"
        fst_period_idx = len(decoded_words)
      sent = decoded_words[:fst_period_idx + 1]  # sentence up to and including the period
      decoded_words = decoded_words[fst_period_idx + 1:]  # everything else
      decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w) for w in reference_sents]

    # Write to file
    ref_file = os.path.join(self._rouge_ref_dir, "reference.txt")
    decoded_file = os.path.join(self._rouge_dec_dir, "decoded.txt")

    with open(ref_file, "a") as f:
      for idx, sent in enumerate(reference_sents):
        f.write(sent + "\n") if idx == len(reference_sents) - 1 else f.write(sent)
    with open(decoded_file, "a") as f:
      for idx, sent in enumerate(decoded_sents):
        #f.write(sent + "\t"+' '.join([str(a) for a in log_probs])+"\n") if idx == len(decoded_sents) - 1 else f.write(sent)
        f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent)
    tf.logging.info("Wrote example %i to file" % ex_index)

  def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens):
    """Write some data to json file, which can be read into the in-browser attention visualizer tool:
      https://github.com/abisee/attn_vis

    Args:
      article: The original article string.
      abstract: The human (correct) abstract string.
      attn_dists: List of arrays; the attention distributions.
      decoded_words: List of strings; the words of the generated summary.
      p_gens: List of scalars; the p_gen values. If not running in pointer-generator mode, list of None.
    """
    article_lst = article.split()  # list of words
    decoded_lst = decoded_words  # list of decoded words
    to_write = {
      'article_lst': [make_html_safe(t) for t in article_lst],
      'decoded_lst': [make_html_safe(t) for t in decoded_lst],
      'abstract_str': make_html_safe(abstract),
      'attn_dists': attn_dists
    }
    if FLAGS.pointer_gen:
      to_write['p_gens'] = p_gens
    output_fname = os.path.join(self._decode_dir, 'attn_vis_data.json')
    with open(output_fname, 'w') as output_file:
      json.dump(to_write, output_file)
    tf.logging.info('Wrote visualization data to %s', output_fname)


def print_results(article, abstract, decoded_output):
  """Prints the article, the reference summmary and the decoded summary to screen"""
  print("---------------------------------------------------------------------------")
  tf.logging.info('ARTICLE:  %s', article)
  tf.logging.info('REFERENCE SUMMARY: %s', abstract)
  tf.logging.info('GENERATED SUMMARY: %s', decoded_output)
  print("---------------------------------------------------------------------------")


def make_html_safe(s):
  """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
  s.replace("<", "&lt;")
  s.replace(">", "&gt;")
  return s


def rouge_eval(ref_dir, dec_dir):
  """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
  r = pyrouge.Rouge155()
  r.model_filename_pattern = '#ID#_reference.txt'
  r.system_filename_pattern = '(\d+)_decoded.txt'
  r.model_dir = ref_dir
  r.system_dir = dec_dir
  logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
  rouge_results = r.convert_and_evaluate()
  return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
  """Log ROUGE results to screen and write to file.

  Args:
    results_dict: the dictionary returned by pyrouge
    dir_to_write: the directory where we will write the results to"""
  log_str = ""
  for x in ["1", "2", "l"]:
    log_str += "\nROUGE-%s:\n" % x
    for y in ["f_score", "recall", "precision"]:
      key = "rouge_%s_%s" % (x, y)
      key_cb = key + "_cb"
      key_ce = key + "_ce"
      val = results_dict[key]
      val_cb = results_dict[key_cb]
      val_ce = results_dict[key_ce]
      log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
  tf.logging.info(log_str)  # log to screen
  results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
  tf.logging.info("Writing final ROUGE results to %s...", results_file)
  with open(results_file, "w") as f:
    f.write(log_str)


def get_decode_dir_name(ckpt_name):
  """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""

  if "train" in FLAGS.data_path:
    dataset = "train"
  elif "dev" in FLAGS.data_path:
    dataset = "dev"
  elif "test" in FLAGS.data_path:
    dataset = "test"
  else:
    raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))

  dataset = FLAGS.dataset
  #if "xuanpin1" in FLAGS.data_path: dataset = "xuanpin1"
  #if "xuanpin2" in FLAGS.data_path: dataset = "xuanpin2"
  #if "xuanpin3" in FLAGS.data_path: dataset = "xuanpin3"
  #if "xuanpin4" in FLAGS.data_path: dataset = "xuanpin4"
  #if "xuanpin5" in FLAGS.data_path: dataset = "xuanpin5"
  #if "xuanpin6" in FLAGS.data_path: dataset = "xuanpin6"
  #if "alphaSales" in FLAGS.data_path: dataset = "alphaSales"

  force_decoding = "_force_decoding" if FLAGS.force_decoding else ""
  ckg = ''
  if "ckg" in FLAGS.ckg_path:
    ckg = "_ckg"
  elif 'analysis' in FLAGS.ckg_path:
    ckg = "_consistent"

  gridBS = "_gridBS" if FLAGS.constraint_path != "" else ""
  sp = ''
  sp = '_0sellPoint' if '0sellingPoints' in FLAGS.constraint_path else sp
  sp = '_1sellPoint' if '1sellingPoints' in FLAGS.constraint_path else sp
  sp = '_2sellPoint' if '2sellingPoints' in FLAGS.constraint_path else sp
  sp = '_3sellPoint' if '3sellingPoints' in FLAGS.constraint_path else sp
  sp = '_4sellPoint' if '4sellingPoints' in FLAGS.constraint_path else sp
  sp = '_5sellPoint' if '5sellingPoints' in FLAGS.constraint_path else sp

  sp_from = ''
  sp_from = '_ocr' if 'ocr' in FLAGS.constraint_path else sp_from
  sp_from = '_title' if 'title' in FLAGS.constraint_path else sp_from
  
  funcWord = '_funcWords' if FLAGS.funcWords_file_path != '' else ''
  dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec%s%s%s%s%s%s_noRedundentUnigram_upgradeConsistent" % (dataset, FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps, force_decoding, ckg, gridBS, sp, sp_from, funcWord)
  
  if ckpt_name is not None:
    dirname += "_%s" % ckpt_name
  return dirname
