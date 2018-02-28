from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class DIVEInputs(object):

  def __init__(self,hps, placeholder):

    batch_size=hps.batch_size
    self.real_batch_size = placeholder['real_batch_size']
    self.row_indices = placeholder['row_indices']
    self.neg_sample_weights = placeholder['neg_sample_weights']
    self.filtering = placeholder['filtering']

class DIVEVariables(object):

  def __init__(self,hps, placeholder):

    self.data_shard = placeholder['data_shard']
    self.data_shard_var = tf.Variable(tf.zeros([hps.data_shard_rows,hps.data_shard_cols], dtype = tf.int32),trainable=False)
    self.assign_data_shard_var = tf.assign(self.data_shard_var,self.data_shard)
    self.word_emb = tf.Variable(tf.random_uniform([hps.vocab_size,hps.emb_size], 0, hps.init_scale), trainable = True, name = 'word_emb')
    self.classifier_emb = tf.Variable(tf.random_uniform([hps.vocab_size,hps.emb_size], 0, hps.init_scale), trainable = True, name = 'classifier_emb')


class DIVEModel(object):

  def nonnegative(self, tensor):
    return tf.maximum(tensor, 0)

  def __init__(self,hps,inputs,variables, unigrams):

    # fixme: this doesnt know how big it's supposed to be

    self.data_shard=variables.data_shard_var

    word_emb=variables.word_emb
    classifier_emb=variables.classifier_emb

    cur_tok_idx_seqs=tf.nn.embedding_lookup(self.data_shard,inputs.row_indices)

    word_emb_seqs_raw = tf.nn.embedding_lookup(word_emb,cur_tok_idx_seqs)
    classifier_emb_seqs_raw = tf.nn.embedding_lookup(classifier_emb,cur_tok_idx_seqs)
    # make them non-negative
    if hps.use_projection:
        word_emb_seqs = word_emb_seqs_raw
        classifier_emb_seqs = classifier_emb_seqs_raw
    else:
        word_emb_seqs = self.nonnegative(word_emb_seqs_raw)
        classifier_emb_seqs = self.nonnegative(classifier_emb_seqs_raw)

    # construct a matrix in shape(batch_size, data_shard_cols, 2 * window_size, emb_size) by tiling and shifting
    skip_gram_mat = tf.Variable(tf.zeros([hps.batch_size, hps.data_shard_cols, 0, hps.emb_size], dtype=tf.float32))
    # left side
    for i in xrange(hps.window_size):
        p = hps.window_size - i
        padding = tf.zeros([hps.batch_size, p, 1, hps.emb_size])
        shifted_seq = tf.concat([padding, tf.expand_dims(classifier_emb_seqs[:, 0: -p, :], 2)], 1)
        skip_gram_mat = tf.concat([skip_gram_mat, shifted_seq], 2)
    for i in xrange(hps.window_size):
        p = i + 1
        padding = tf.zeros([hps.batch_size, p, 1, hps.emb_size])
        shifted_seq = tf.concat([tf.expand_dims(classifier_emb_seqs[:, p: , :], 2), padding], 1)
        skip_gram_mat = tf.concat([skip_gram_mat, shifted_seq], 2)
    # multiply cbow with filtering matrix
    skip_gram_mat = skip_gram_mat * tf.expand_dims(inputs.filtering, 3)

    # skip_gram objective
    pos_dots = tf.matmul(skip_gram_mat, tf.expand_dims(word_emb_seqs, 3))
    pos_obj = tf.reduce_sum(tf.log(tf.sigmoid(pos_dots)))

    # negative sampling:
    sampled_rows, _, _ = tf.nn.uniform_candidate_sampler(
            tf.expand_dims(inputs.row_indices, 0),
            hps.batch_size, hps.batch_size * hps.num_neg_samples, False, hps.data_shard_rows)       # (batch_size * num_neg_samples)
    sampled_index_seq = tf.nn.embedding_lookup(self.data_shard, 
            tf.reshape(sampled_rows, [hps.batch_size, hps.num_neg_samples]))                        # (batch_size, num_neg_samples, data_shard_cols)
    sampled_index_seq = tf.transpose(sampled_index_seq, [2, 0, 1])                                  # (data_shard_cols, batch_size, num_neg_samples)
    shuffled_sampled_index_seq = tf.transpose(tf.random_shuffle(sampled_index_seq), [1, 0, 2])      # (batch_size, data_shard_cols, num_neg_samples)
    neg_classifier_emb_seqs = tf.nn.embedding_lookup(classifier_emb, shuffled_sampled_index_seq)    # (batch_size, data_shard_cols, num_neg_samples, emb_size)
    if not hps.use_projection: neg_classifier_emb_seqs = tf.nonnegative(neg_classifier_emb_seqs)

    neg_dots = tf.matmul(neg_classifier_emb_seqs, tf.expand_dims(word_emb_seqs, 3))
    neg_logprob = tf.log(tf.sigmoid(-neg_dots))
    # penalize negative part so that general tokens don't get pulled too much
    neg_decayed_term = tf.reshape(neg_logprob, [hps.batch_size, hps.data_shard_cols, hps.num_neg_samples]) \
        * tf.expand_dims(inputs.neg_sample_weights, 2)
    neg_obj = tf.reduce_sum(neg_decayed_term)
    
    self.loss = -(pos_obj + neg_obj) / tf.to_float(inputs.real_batch_size*hps.num_neg_samples*hps.data_shard_cols)

    self.convert_we_op = tf.assign(variables.word_emb, self.nonnegative(word_emb))
    self.convert_ce_op = tf.assign(variables.classifier_emb, self.nonnegative(classifier_emb))

  def loss(self):
    return self.loss

  def training(self, loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


