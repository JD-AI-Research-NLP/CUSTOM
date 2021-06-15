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
#KB2.0 limitations under the License.
# ==============================================================================

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS

class SummarizationModel(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, hps, vocab, vocab_aspect):
    self._hps = hps
    self.lr = hps.lr
    self._vocab = vocab
    self._vocab_aspect = vocab_aspect
    sp_word_mask = [1] * self._vocab.size()
    r1 = open(FLAGS.sp_word_mask, 'r')
    for line in r1.readlines():
       word = line.strip()
       if word in vocab._word_to_id:
           sp_word_mask[vocab._word_to_id[word]] = 1e-10
    self.sp_word_mask = sp_word_mask
   

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps
    # encoder part
    self._enc_batch_caption = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_caption')
    self._enc_batch_aspect = tf.placeholder(tf.int32, [hps.batch_size], name='enc_batch_aspect')
    self._enc_batch_attribute = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_attribute')
    self._enc_batch_value = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_value')

    self._enc_table_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_table_lens')
    self._enc_caption_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_caption_lens')
    self._enc_caption_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_caption_padding_mask')
    self._enc_table_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_table_padding_mask')
    # selector part
    self._art_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_art_len, hps.max_sent_len], name='art_batch')
    self._art_lens = tf.placeholder(tf.int32, [hps.batch_size], name='art_lens')
    self._sent_lens = tf.placeholder(tf.int32, [hps.batch_size, hps.max_art_len], name='sent_lens')
    self._focus_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_art_len], name='sent_lens')

    self._art_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_art_len], name='art_padding_mask')
    self._sent_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_art_len, hps.max_sent_len], name='sent_padding_mask')
    self._enc_sent_id_mask = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_sent_id_mask')
    
    if FLAGS.pointer_gen:
      self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
      self._enc_attribute_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_attribute_batch_extend_vocab')
      self._enc_value_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_value_batch_extend_vocab')
      self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

    # decoder part
    self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
    self._constraint_id = tf.placeholder(tf.int32, [hps.batch_size], name='constraint_id')
    self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
    self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')

    if hps.mode=="decode" and hps.coverage:
      self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')

  def _make_feed_dict(self, batch, just_enc=False):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

    Args:
      batch: Batch object
      just_enc: Boolean. If True, only feed the parts needed for the encoder.
    """
    feed_dict = {}
    feed_dict[self._enc_batch_caption] = batch.enc_batch
    feed_dict[self._enc_batch_aspect] = batch.enc_batch_aspect
    feed_dict[self._enc_batch_attribute] = batch.enc_batch_attribute
    feed_dict[self._enc_batch_value] = batch.enc_batch_value
    feed_dict[self._enc_caption_lens] = batch.enc_lens
    feed_dict[self._enc_table_lens] = batch.enc_table_lens
    feed_dict[self._enc_caption_padding_mask] = batch.enc_padding_mask
    feed_dict[self._enc_table_padding_mask] = batch.enc_table_padding_mask
    # selector part
    feed_dict[self._art_batch] = batch.art_batch # (batch_size, max_art_len, max_sent_len)
    feed_dict[self._art_lens] = batch.art_lens   # (batch_size, )
    feed_dict[self._sent_lens] = batch.sent_lens # (batch_size, max_art_len)
    feed_dict[self._focus_batch] = batch.focus_batch # (batch_size, max_art_len)
    feed_dict[self._art_padding_mask] = batch.art_padding_mask # (batch_size, max_art_len)
    feed_dict[self._sent_padding_mask] = batch.sent_padding_mask # (batch_size, max_art_lens, max_sent_len)
    feed_dict[self._enc_sent_id_mask] = batch.enc_sent_id_mask
    if FLAGS.pointer_gen:
      feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed_dict[self._enc_attribute_batch_extend_vocab] = batch.enc_attribute_batch_extend_vocab
      feed_dict[self._enc_value_batch_extend_vocab] = batch.enc_value_batch_extend_vocab
      feed_dict[self._max_art_oovs] = batch.max_art_oovs
    if not just_enc:
      feed_dict[self._dec_batch] = batch.dec_batch
      feed_dict[self._target_batch] = batch.target_batch
      feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
    return feed_dict

  def _add_table_encoder(self, encoder_inputs, table_len, caption_len):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    attribute_inputs, value_inputs, caption_inputs = encoder_inputs
    with tf.variable_scope('table_encoder'):
      cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      (encoder_caption_outputs, (fw_st, bw_st)) = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, caption_inputs, dtype=tf.float32,
                                        sequence_length=caption_len, swap_memory=True)
      encoder_caption_outputs = tf.concat(axis=2, values=encoder_caption_outputs) # concatenate the forwards and backwards states
      encoder_table_outputs = tf.concat(axis=2, values=(attribute_inputs, value_inputs))
      attribute_inputs = tf.layers.dense(inputs=attribute_inputs, units=self._hps.hidden_dim * 2)
      encoder_table_outputs = tf.layers.dense(inputs=encoder_table_outputs, units=self._hps.hidden_dim * 2)
      encoder_outputs = tf.concat(axis=1, values = (encoder_caption_outputs, attribute_inputs, encoder_table_outputs))
      #table_mask = tf.cast(tf.sequence_mask(table_len,maxlen=None), dtype=tf.float32) # max length in attr_len if maxlen=None
      # encoder_table_outputs = encoder_table_outputs * tf.expand_dims(table_mask, axis=-1)
      #table_final_state = tf.reduce_sum(encoder_table_outputs, axis=1)/tf.expand_dims(table_len, axis=-1)

      #encoder_outputs = tf.concat(axis=1, values=(encoder_caption_outputs, encoder_table_outputs))

      #w = tf.get_variable('w', [self._hps.emb_dim, self._hps.hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      #v = tf.get_variable('v', [self._hps.hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      #encoder_table_inputs = tf.nn.xw_plus_b(table_inputs, w, v)
    return encoder_outputs, fw_st, bw_st

    #return encoder_caption_outputs,encoder_table_outputs， fw_st, bw_st

  def _add_encoder(self, encoder_inputs, seq_len):
    """Add a single-layer bidirectional LSTM encoder to the graph.
    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].
    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.variable_scope('encoder'):
      cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states
    return encoder_outputs, fw_st, bw_st

  def _add_selector_encoder(self, encoder_inputs, seq_len, name):
    with tf.variable_scope(name):
      cell_fw = tf.contrib.rnn.GRUCell(self._hps.hidden_dim_selector)
      cell_bw = tf.contrib.rnn.GRUCell(self._hps.hidden_dim_selector)
      (encoder_outputs, (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states
    return encoder_outputs

  def _reduce_states(self, fw_st, bw_st):
    """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

    Args:
      fw_st: LSTMStateTuple with hidden_dim units.
      bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    hidden_dim = self._hps.hidden_dim
    with tf.variable_scope('reduce_final_st'):

      # Define weights and biases to reduce the cell and reduce the state
      w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

      # Apply linear layer
      old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c]) # Concatenation of fw and bw cell
      old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h]) # Concatenation of fw and bw state
      new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
      new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
      return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state

  def _add_decoder(self, inputs):
    """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

    Args:
      inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

    Returns:
      outputs: List of tensors; the outputs of the decoder
      out_state: The final state of the decoder
      attn_dists: A list of tensors; the attention distributions
      p_gens: A list of scalar tensors; the generation probabilities
      coverage: A tensor, the current coverage vector
    """
    hps = self._hps
    cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

    prev_coverage = self.prev_coverage if hps.mode=="decode" and hps.coverage else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time
    selector_probs = self._selector_probs
    enc_sent_id_mask = self._enc_sent_id_mask
    outputs, out_state, attn_dists, attn_dists_norescale, p_gens, coverage = \
      attention_decoder(inputs, self._dec_in_state,
                        self._enc_states,
                        self._enc_padding_mask, self._enc_aspect, cell,
                        initial_state_attention=(hps.mode=="decode"),
                        pointer_gen=hps.pointer_gen,
                        use_coverage=hps.coverage,
                        prev_coverage=prev_coverage,
                        selector_probs=selector_probs,
                        enc_sent_id_mask=enc_sent_id_mask)

    return outputs, out_state, attn_dists, p_gens, coverage

  def _add_table_decoder_back(self, inputs):
    """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

    Args:
      inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

    Returns:
      outputs: List of tensors; the outputs of the decoder
      out_state: The final state of the decoder
      attn_dists: A list of tensors; the attention distributions
      p_gens: A list of scalar tensors; the generation probabilities
      coverage: A tensor, the current coverage vector
    """
    hps = self._hps
    cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

    prev_coverage = self.prev_coverage if hps.mode=="decode" and hps.coverage else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time

    outputs, out_state, attn_dists, p_gens, coverage = \
      attention_decoder(inputs,
                        self._dec_in_state,
                        self._enc_table_states,
                        self._enc_table_padding_mask,
                        self._enc_caption_states,
                        self._enc_caption_padding_mask,
                        cell,
                        initial_state_attention=(hps.mode=="decode"),
                        pointer_gen=hps.pointer_gen,
                        use_coverage=hps.coverage,
                        prev_coverage=prev_coverage)

    return outputs, out_state, attn_dists, p_gens, coverage

  def _calc_final_dist(self, vocab_dists, attn_dists, sp_word_mask):
    """Calculate the final distribution, for the pointer-generator model

    Args:
      vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
      attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

    Returns:
      final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    with tf.variable_scope('final_distribution'):
      # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
      print(vocab_dists)
      self.vocab_dists_ = vocab_dists
      vocab_dists = [p_gen * dist for (p_gen,dist) in zip(self.p_gens, vocab_dists)]
      #vocab_dists = [dist * sp_word_mask for (p_gen,dist) in zip(self.p_gens, vocab_dists)]
      self.vocab_dists__ = vocab_dists
      attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(self.p_gens, attn_dists)]

      # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
      extended_vsize = self._vocab.size() + self._max_art_oovs # the maximum (over the batch) size of the extended vocabulary
      extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
      vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists] # list length max_dec_steps of shape (batch_size, extended_vsize)
      #final_dists = [vocab_dist for vocab_dist in vocab_dists_extended]
      #return final_dists
      # Project the values in the attention distributions onto the appropriate entries in the final distributions
      # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
      # This is done for each decoder timestep.
      # This is fiddly; we use tf.scatter_nd to do the projection
      batch_nums = tf.range(0, limit=self._hps.batch_size) # shape (batch_size)
      batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
      attn_len = tf.shape(self._enc_batch_extend_vocab)[1] # number of states we attend over
#       attn_attribute_len = tf.shape(self._enc_attribute_batch_extend_vocab)[1] # number of states we attend over
#       attn_value_len = tf.shape(self._enc_value_batch_extend_vocab)[1] # number of states we attend over
#       batch_nums = tf.tile(batch_nums, [1, attn_len + attn_attribute_len + attn_value_len]) # shape (batch_size, attn_len)
      batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
#       indices = tf.stack((batch_nums,
#                           tf.concat(axis=1, values=(self._enc_batch_extend_vocab,
#                                      self._enc_attribute_batch_extend_vocab,
#                                      self._enc_value_batch_extend_vocab))),
#                           axis=2) # shape (batch_size, enc_t, 2)
      indices = tf.stack((batch_nums, self._enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
      shape = [self._hps.batch_size, extended_vsize]
      attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] # list length max_dec_steps (batch_size, extended_vsize)

      # Add the vocab distributions and the copy distributions together to get the final distributions
      # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
      # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
      final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]
      #print(final_dists)
      return final_dists

  def _add_emb_vis(self, embedding_var):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    train_dir = os.path.join(FLAGS.log_root, "train")
    vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
    self._vocab.write_metadata(vocab_metadata_path) # write metadata file
    summary_writer = tf.summary.FileWriter(train_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = vocab_metadata_path
    projector.visualize_embeddings(summary_writer, config)

  def _add_table2seq(self):
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    vsize = self._vocab.size() # size of the vocabulary
    vsize_aspect = self._vocab_aspect.size()
    with tf.variable_scope('table2seq'):
      # Some initializers
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)
      # Add embedding matrix (shared by the encoder and decoder inputs)
      with tf.variable_scope('embedding'):
        embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        embedding_aspect = tf.get_variable('embedding_aspect', [vsize_aspect, hps.aspect_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        if hps.mode=="train": self._add_emb_vis(embedding) # add to tensorboard
        emb_enc_attr_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch_attribute) # tensor with shape (batch_size, max_enc_steps, emb_size)
        emb_enc_value_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch_value) # tensor with shape (batch_size, max_enc_steps, emb_size)
        emb_enc_caption_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch_caption) # tensor with shape (batch_size, max_enc_steps, emb_size)
        emb_aspect_inputs = tf.nn.embedding_lookup(embedding_aspect, self._enc_batch_aspect)
        emb_enc_caption_inputs = emb_enc_caption_inputs + tf.tile(tf.expand_dims(emb_aspect_inputs, 1), [1, tf.shape(emb_enc_caption_inputs)[1], 1])
        emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)
        # 增加sentence级别的embedding
        emb_batch = tf.nn.embedding_lookup(embedding, self._art_batch)

      # Add selector encoder
      sent_enc_inputs = tf.reshape(emb_batch, [-1, hps.max_sent_len, hps.emb_dim]) # (batch_size*max_art_len, max_sent_len, emb_dim)
      sent_lens = tf.reshape(self._sent_lens, [-1]) # (batch_size*max_art_len, )
      sent_enc_outputs = self._add_selector_encoder(sent_enc_inputs, sent_lens, name='sent_encoder') # (batch_size*max_art_len, max_sent_len, hidden_dim*2)
      # Add sentence-level encoder to produce sentence representations.
      # sentence-level encoder input: average-pooled, concatenated hidden states of the word-level bi-LSTM.
      sent_padding_mask = tf.reshape(self._sent_padding_mask, [-1, hps.max_sent_len, 1]) # (batch_size*max_art_len, max_sent_len, 1)
      sent_lens_float = tf.reduce_sum(sent_padding_mask, axis=1)
      self.sent_lens_float = tf.where(sent_lens_float > 0.0, sent_lens_float, tf.ones(sent_lens_float.get_shape().as_list()))
      art_enc_inputs = tf.reduce_sum(sent_enc_outputs * sent_padding_mask, axis=1) / self.sent_lens_float # (batch_size*max_art_len, hidden_dim*2)
      art_enc_inputs = tf.reshape(art_enc_inputs, [hps.batch_size, -1, hps.hidden_dim_selector*2]) # (batch_size, max_art_len, hidden_dim*2)
      art_enc_outputs = self._add_selector_encoder(art_enc_inputs, self._art_lens, name='art_encoder') # (batch_size, max_art_len, hidden_dim*2)
      # Get each sentence representation and the document representation.
      sent_feats = tf.contrib.layers.fully_connected(art_enc_outputs, hps.aspect_dim, activation_fn=tf.tanh) # (batch_size, max_art_len, hidden_dim)
      # 计算每个句子的得分，利用Aspect的embedding进行计算
#       emb_aspect_expand = tf.tile(tf.expand_dims(emb_aspect_inputs, 1), [1, hps.max_art_len, 1])  # (batch_size, max_art_len, hidden)
      emb_aspect_trans = tf.expand_dims(emb_aspect_inputs, 2)
      logits = tf.squeeze(tf.matmul(sent_feats, emb_aspect_trans), 2)# (batch_size, max_art_len)
      probs = tf.sigmoid(logits)
      self._selector_probs = probs * self._art_padding_mask
      tf.summary.histogram('selector_probs', self._selector_probs[0])
      losses_extractor = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(self._focus_batch, tf.float32)) # (batch_size, max_art_len)
      loss_extractor = tf.reduce_sum(losses_extractor * self._art_padding_mask, 1) / tf.reduce_sum(self._art_padding_mask, 1) # (batch_size,)
      self._loss_extractor = tf.reduce_mean(loss_extractor)
      
#       art_padding_mask = tf.expand_dims(self._art_padding_mask, 2) # (batch_size, max_art_len, 1)
#       art_feats = tf.reduce_sum(art_enc_outputs * art_padding_mask, axis=1) / tf.reduce_sum(art_padding_mask, axis=1) # (batch_size, hidden_dim)
#       art_feats = tf.contrib.layers.fully_connected(art_feats, hps.hidden_dim_selector, activation_fn=tf.tanh) # (batch_size, hidden_dim)
        
      # Add the encoder.
      enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_caption_inputs, self._enc_caption_lens)
      self._enc_states = enc_outputs
      self._enc_aspect = emb_aspect_inputs
      # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
      self._dec_in_state = self._reduce_states(fw_st, bw_st)
      # Add the decoder.
      with tf.variable_scope('decoder'):
#         self._enc_padding_mask = tf.concat(axis=1, values=(self._enc_caption_padding_mask, self._enc_table_padding_mask, self._enc_table_padding_mask))
        self._enc_padding_mask = self._enc_caption_padding_mask
        decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = \
          self._add_decoder(emb_dec_inputs)

      # Add the output projection to obtain the vocabulary distribution
      with tf.variable_scope('output_projection'):
        w = tf.get_variable('w', [hps.hidden_dim, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
        w_t = tf.transpose(w)
        v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
        vocab_scores = [] # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
        for i,output in enumerate(decoder_outputs):
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          vocab_scores.append(tf.nn.xw_plus_b(output, w, v)) # apply the linear layer

        vocab_dists = [tf.nn.softmax(s) for s in vocab_scores] # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.


      # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
      if FLAGS.pointer_gen:
        final_dists = self._calc_final_dist(vocab_dists, self.attn_dists, self.sp_word_mask)
      else: # final distribution is just vocabulary distribution
        final_dists = vocab_dists
      if hps.mode in ['train', 'eval']:
        # Calculate the loss
        with tf.variable_scope('loss'):
          if FLAGS.pointer_gen:
            # Calculate the loss per step
            # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
            loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
            batch_nums = tf.range(0, limit=hps.batch_size) # shape (batch_size)
            for dec_step, dist in enumerate(final_dists):
              targets = self._target_batch[:,dec_step] # The indices of the target words. shape (batch_size)
              indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
              gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
              losses = -tf.log(gold_probs)
              loss_per_step.append(losses)

            # Apply dec_padding_mask and get loss
            self._loss_word = _mask_and_avg(loss_per_step, self._dec_padding_mask)
            self._loss = self._loss_word + hps.selector_loss_wt * self._loss_extractor

          else: # baseline model
            self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self._target_batch, self._dec_padding_mask) # this applies softmax internally
          tf.summary.scalar('loss_extractor', self._loss_extractor)
          self.loss_summary = tf.summary.scalar('loss', self._loss)

          # Calculate coverage loss from the attention distributions
          if hps.coverage:
            with tf.variable_scope('coverage_loss'):
              self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
              tf.summary.scalar('coverage_loss', self._coverage_loss)
            self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
            tf.summary.scalar('total_loss', self._total_loss)

    if hps.mode == "decode":
      # We run decode beam search mode one decoder step at a time
      assert len(final_dists)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
      final_dists = final_dists[0]
      topk_probs, self._topk_ids = tf.nn.top_k(final_dists, 100) # take the k largest probs. note batch_size=beam_size in decode mode
      self._topk_log_probs = tf.log(topk_probs + 0.0000001)
      self._final_dist = final_dists
      indices=tf.range(0, limit=hps.batch_size)
      indices=tf.stack((indices,self._constraint_id), axis=1)
      self.constraint_prob = tf.log(tf.gather_nd(final_dists,indices) + 0.0000001)
      #print(self.constraint_prob)

  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    with tf.device("/gpu:0"):
      grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(self.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
    with tf.device("/gpu:0"):
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    self._add_placeholders()
    with tf.device("/gpu:0"):
      self._add_table2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)

  def run_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'train_op': self._train_op,
        'summaries': self._summaries,
        'loss': self._loss,
        'loss_word': self._loss_word,
        'loss_ext': self._loss_extractor,
        'global_step': self.global_step,
    }
    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss
    return sess.run(to_return, feed_dict)

  def run_eval_step(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'summaries': self.loss_summary,
        'loss': self._loss,
        'loss_word': self._loss_word,
        'loss_ext': self._loss_extractor
#         'global_step': self.global_step,
    }
    return sess.run(to_return, feed_dict)

  def run_encoder(self, sess, batch):
    """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

    Args:
      sess: Tensorflow session.
      batch: Batch object that is the same example repeated across the batch (for beam search)

    Returns:
      enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
      dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
    """
    
    feed_dict = self._make_feed_dict(batch, just_enc=True) # feed the batch into the placeholders
    (enc_states, enc_aspect, dec_in_state, selector_probs, global_step) = sess.run([self._enc_states, self._enc_aspect, self._dec_in_state,  self._selector_probs, self.global_step], feed_dict) # run the encoder

    # dec_in_state is LSTMStateTuplebatch_size,hidden_dim],[batch_size,hidden_dim])
    # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
    dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
    return enc_states, enc_aspect, dec_in_state, selector_probs

  def decode_onestep_constraint(self, sess, batch, latest_tokens, constraint_tokens,
                                enc_states, enc_aspects, dec_init_states, selector_probs, prev_coverage):
    beam_size = len(dec_init_states)
#     import pdb
#     pdb.set_trace()
    # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
    cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
    hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
    new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
    new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
    new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
    feed = {
      self._enc_states: enc_states,
      self._enc_aspect: enc_aspects,
      self._selector_probs: selector_probs,
      self._enc_caption_padding_mask: batch.enc_padding_mask,
      self._enc_table_padding_mask: batch.enc_table_padding_mask,
      self._enc_sent_id_mask: batch.enc_sent_id_mask,
      self._dec_in_state: new_dec_in_state,
      self._dec_batch: np.transpose(np.array([latest_tokens])),
      self._constraint_id: constraint_tokens
    }
    to_return = {
      "pro_1": self.vocab_dists_,
      "pro_2": self.vocab_dists__,
      "ids": self._topk_ids,
      "probs": self._topk_log_probs,
      "states": self._dec_out_state,
      "attn_dists": self.attn_dists,
      "constraint_prob": self.constraint_prob
    }
    if FLAGS.pointer_gen:
      feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed[self._enc_attribute_batch_extend_vocab] = batch.enc_attribute_batch_extend_vocab
      feed[self._enc_value_batch_extend_vocab] = batch.enc_value_batch_extend_vocab
      feed[self._max_art_oovs] = batch.max_art_oovs
      to_return['p_gens'] = self.p_gens
    #print(batch.enc_batch_extend_vocab)
    #print(batch.enc_attribute_batch_extend_vocab)
    #print(batch.enc_value_batch_extend_vocab)
    #print(batch.max_art_oovs)
    if self._hps.coverage:
      feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
      to_return['coverage'] = self.coverage

    results = sess.run(to_return, feed_dict=feed)  # run the decoder step
    #print(results['states'])
    # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
    new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in
                  range(beam_size)]

    # Convert singleton list containing a tensor to a list of k arrays
    assert len(results['attn_dists']) == 1
    attn_dists = results['attn_dists'][0].tolist()

    if FLAGS.pointer_gen:
      # Convert singleton list containing a tensor to a list of k arrays
      assert len(results['p_gens']) == 1
      p_gens = results['p_gens'][0].tolist()
    else:
      p_gens = [None for _ in range(beam_size)]

    # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
    if FLAGS.coverage:
      new_coverage = results['coverage'].tolist()
      assert len(new_coverage) == beam_size
    else:
      new_coverage = [None for _ in range(beam_size)]
    #print(results['pro_1'][0][0][5])
    #print(results['pro_2'][0][0][5])
    return results['ids'], results['probs'], results['constraint_prob'], new_states, attn_dists, p_gens, new_coverage

def _mask_and_avg(values, padding_mask):
  """Applies mask to values then returns overall average (a scalar)

  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

  Returns:
    a scalar
  """

  dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
  values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
  values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
  return tf.reduce_mean(values_per_ex) # overall average

def _coverage_loss(attn_dists, padding_mask):
  """Calculates the coverage loss from the attention distributions.

  Args:
    attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).

  Returns:
    coverage_loss: scalar
  """
  coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  for a in attn_dists:
    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
    covlosses.append(covloss)
    coverage += a # update the coverage vector
  coverage_loss = _mask_and_avg(covlosses, padding_mask)
  return coverage_loss
