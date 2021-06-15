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

"""This file contains some utility functions"""

import tensorflow as tf
import time
import os
import re

FLAGS = tf.app.flags.FLAGS

def get_config():
  """Returns config for tf.session"""
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  return config

#def load_ckpt(saver, sess, ckpt_dir="train"):

#def load_ckpt(saver, sess, ckpt_dir="train_561730"):
  
#def load_ckpt(saver, sess, ckpt_dir="train_502813"):
#def load_ckpt(saver, sess, ckpt_dir="train_1004420"):
#def load_ckpt(saver, sess, ckpt_dir="train_1864240"):

#def load_ckpt(saver, sess, ckpt_dir="train_396490"):
#def load_ckpt(saver, sess, ckpt_dir="train_616206"):
#def load_ckpt(saver, sess, ckpt_dir="train_764986"):

def load_ckpt(saver, sess, ckpt_dir="train"):
  """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
  while True:
      latest_filename = "checkpoint" if ckpt_dir=="eval" else None
      ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
      #print(ckpt_dir)
      ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
      #tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
      saver.restore(sess, ckpt_state.model_checkpoint_path)
      return ckpt_state.model_checkpoint_path

def join_tokens(tokens):
    pattern = re.compile(r'[^\u4e00-\u9fa5，。？；！、]+$')
    tmp = ''
    for i in range(len(tokens)):
        if i > 0 and re.match(pattern, tokens[i-1] + ' ' + tokens[i]):
            tmp = tmp + ' ' + tokens[i]
        else:
            tmp = tmp + tokens[i]
    return tmp.strip()

def conflit_unit(s):
    valuePattern = re.compile(r'[^\u4e00-\u9fa5，。？；！、 ]+')
    longValuePattern = re.compile(r'[^\u4e00-\u9fa5，。？；！、]+')
    valueUnitPattern = re.compile(r'([^\u4e00-\u9fa5，。？；！、 ]+)([\u4e00-\u9fa5])')
    
#     pattern = r'([^\u4e00-\u9fa5，。？；！、]+)([\u4e00-\u9fa5])'
    inputSeq = s.strip().lower()
    values = [_.strip() for _ in re.findall(valuePattern, inputSeq) if _.strip() != '']
    longValues = [_.strip() for _ in re.findall(longValuePattern, inputSeq) if _.strip() != '']
    valueUnits = re.findall(valueUnitPattern, inputSeq)
    
#     ms = re.findall(pattern, s.strip().lower())

    res = {}
    for num, unit in valueUnits:
        if unit not in res:
            res[unit] = set()
        res[unit].add(num)
    
    confVUs = [[_ + k for _ in v] for k,v in res.items() if len(v) > 1]
    confVUs = [_ for items in confVUs for _ in items]
    valueUnits = [[_ + k for _ in v] for k,v in res.items()]
    valueUnits = [_ for items in valueUnits for _ in items]

    return confVUs, values, longValues, valueUnits