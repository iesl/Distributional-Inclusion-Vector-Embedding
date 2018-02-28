from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import math
import json
import random
import msgpack
import numpy as np
import tensorflow as tf
import preprocessing_utils as utils
import DIVE_model as DIVE

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 128, 'Batch size. Must divide evenly into the dataset sizes.') #256
flags.DEFINE_integer('vocab_size', 80000, 'vocab_size')
flags.DEFINE_float('init_scale', 0.1, 'init_scale')
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
flags.DEFINE_float('epsilon', 1e-6, 'epsilon')
flags.DEFINE_float('max_grad_norm', 10.0, 'max_grad_norm')
flags.DEFINE_integer('num_neg_samples', 30, 'num_neg_samples')
flags.DEFINE_integer("window_size", 10, "The number of words to predict to the left and right ")
flags.DEFINE_integer('emb_size', 100, 'word embedding dimension')
flags.DEFINE_integer('data_shard_rows', 256*2000, 'num text "lines" for training in one shard') #256*600
flags.DEFINE_integer('data_shard_cols', 100, 'num tokens per text line') #100
flags.DEFINE_integer('max_epoch', 15, 'max_epoch')
flags.DEFINE_float('positive_weight', 1, 'multiply the weights to the loss function of positive samples')
flags.DEFINE_string('log_dir', 'logs', 'logdir path where model is saved')
flags.DEFINE_string('training_file', '', 'training corpus filepath')
flags.DEFINE_boolean('use_projection', False, 'use projection or not')
flags.DEFINE_boolean('use_neg_weight', True, 'use inverse freq as negative weighting or not')

def multiply_positive_sample_weight(PMI_filters, positive_weight):
    for row_ind in range(len(PMI_filters)):
        for col_ind in range(len(PMI_filters[row_ind])):
            for win_ind in range(len(PMI_filters[row_ind][col_ind])):
                PMI_filters[row_ind][col_ind][win_ind] = positive_weight * PMI_filters[row_ind][col_ind][win_ind]    
    return PMI_filters

def placeholder_inputs(FLAGS):
    placeholder = {}
    placeholder['row_indices'] = tf.placeholder(tf.int64, shape = [FLAGS.batch_size])
    placeholder['real_batch_size'] = tf.placeholder(tf.int32, shape = [])
    placeholder['data_shard'] = tf.placeholder(tf.int32,shape=[FLAGS.data_shard_rows, FLAGS.data_shard_cols])
    placeholder['neg_sample_weights'] = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.data_shard_cols])
    placeholder['filtering'] = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.data_shard_cols, 2 * FLAGS.window_size])
    return placeholder

def fill_feed_dict(placeholder, row_indices, neg_sample_weights, filtering):
    feed_dict = {
                placeholder['row_indices']: row_indices, 
                placeholder['real_batch_size']: len(row_indices), 
                placeholder['neg_sample_weights']: neg_sample_weights,
                placeholder['filtering']: filtering
                }
    return feed_dict

def partial_fill_feed_dict(placeholder, data_shared):
    feed_dict = {
                placeholder['data_shard']: data_shared, 
                }
    return feed_dict

def compute_neg_weights(data, row_indices, wcount, use_neg_weight):
    Z = float(FLAGS.data_shard_rows * FLAGS.data_shard_cols) / FLAGS.vocab_size # this is used to ensure avg weights = 1
    weights = []
    for i in row_indices:
        row_weights = []
        for word in data[i, :]:
            if use_neg_weight:
                if wcount[word][1] == 0: row_weights.append(0)
                else: row_weights.append(1.0 / wcount[word][1] * Z)
            else:
                row_weights.append(1.0)
        weights.append(row_weights)
    return weights

def load_PMI_filter_weights(row_indices, PMI_filters):
    weights = []
    for i in row_indices:
        weights.append(PMI_filters[i])
    return weights

def save_word_emb(file_path, emb_var, wcount):
    formatted_emb = {}
    for i, item in enumerate(wcount):
        emb = {}
        for dim, v in enumerate(emb_var[i, :]):
            v = float(v)
            if v != 0: emb[str(dim)] = v
        formatted_emb[item[0]] = emb
    open(file_path, 'w').write(json.dumps(formatted_emb, indent=4, separators=(',', ': ')))

def main(unused_args):
    print("Start!")
    print("use_neg_weight"+str(FLAGS.use_neg_weight))
    if not os.path.exists(FLAGS.log_dir): os.makedirs(FLAGS.log_dir)

    print("Build dataset...")
    data, count, dictionary, reverse_dictionary, PMI_filters = msgpack.unpackb(open(FLAGS.training_file).read())
    FLAGS.vocab_size = len(dictionary)
    print("Vocab size"+str(FLAGS.vocab_size))
    ideal_row_num = (FLAGS.data_shard_rows // FLAGS.batch_size)*FLAGS.batch_size
    if ideal_row_num != FLAGS.data_shard_rows:
        print("new row number",ideal_row_num)
        data = data[:ideal_row_num*FLAGS.data_shard_cols]
        PMI_filters = PMI_filters[:ideal_row_num]
        FLAGS.data_shard_rows = ideal_row_num
    PMI_filters = multiply_positive_sample_weight(PMI_filters, FLAGS.positive_weight)
    

    print("Setup model...")
    data_shared = np.array(data).reshape((FLAGS.data_shard_rows, FLAGS.data_shard_cols))
    placeholder = placeholder_inputs(FLAGS)
    DIVE_inputs = DIVE.DIVEInputs(FLAGS, placeholder)
    DIVE_vars = DIVE.DIVEVariables(FLAGS, placeholder)
        
    unigram_count = [x[1] for x in count]
    DIVE_model = DIVE.DIVEModel(FLAGS, DIVE_inputs, DIVE_vars, unigram_count)
    # Build loss function
    loss = DIVE_model.loss
    # Build Training Function
    train_op = DIVE_model.training(loss, FLAGS.learning_rate)

    with tf.Session() as session, session.as_default():
        
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        session.run(init)
        partial_feed_dict = partial_fill_feed_dict(placeholder, data_shared)
        session.run(DIVE_vars.assign_data_shard_var, feed_dict = partial_feed_dict)
        
        print("Start training...")
        t0 = time.time()
        for cur_epoch in range(FLAGS.max_epoch):
            start, loss_value = 0, 0.0
            perm = np.arange(FLAGS.data_shard_rows)
            random.shuffle(perm)
            while start < FLAGS.data_shard_rows:
                row_indices = perm[start: start + FLAGS.batch_size if start + FLAGS.batch_size < FLAGS.data_shard_rows else FLAGS.data_shard_rows]
                start += FLAGS.batch_size
                neg_weights = compute_neg_weights(data_shared, row_indices, count, FLAGS.use_neg_weight)
                filter_weights = np.array(load_PMI_filter_weights(row_indices, PMI_filters))
                feed_dict = fill_feed_dict(placeholder, row_indices, neg_weights, filter_weights)
                _, step_loss = session.run([train_op, loss], feed_dict=feed_dict)
                loss_value += step_loss
                if FLAGS.use_projection:
                    session.run(DIVE_model.convert_we_op)
                    session.run(DIVE_model.convert_ce_op)

            msg_to_send = '[%s] epoch %d loss %.6f speed: %.0f s' % (FLAGS.log_dir, cur_epoch, loss_value, (time.time() - t0))
            print(msg_to_send)
    		

        session.run(DIVE_model.convert_we_op)
        session.run(DIVE_model.convert_ce_op)
        saver.save(session, FLAGS.log_dir + '/DIVE-model')
        save_word_emb(FLAGS.log_dir + '/word-emb.json', session.run(DIVE_vars.word_emb), count)
        save_word_emb(FLAGS.log_dir + '/context-emb.json', session.run(DIVE_vars.classifier_emb), count)



if __name__ == "__main__":
  tf.app.run()
