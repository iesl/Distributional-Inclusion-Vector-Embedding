import json, msgpack, sys, getopt
import preprocessing_utils as utils
import numpy as np


class Config(object):
    def __init__(self):
        #self.training_file = 'data/wackypedia_512k_matrix_wostop.gz'
        #self.training_file = 'data/wackypedia_512k_POS0_wostop.gz'
        #self.output_coocur_file = 'data/wiki_512k_wostop_coocur_pmi_w_10_th10'
        #self.output_coocur_file = 'data/wiki_512k_POS_wostop_coocur_pmi_w_10_th10'
        #self.output_training_file = 'data/wiki_512k_wostop_pmi_w_10_neg30_th10'
        #self.output_training_file = 'data/wiki_512k_POS_wostop_pmi_w_10_neg30_th10'
        #self.vocab_size = 80000
        self.vocab_size = -10
        #self.data_shard_rows = 512 * 1000
        #self.data_shard_rows = 289560
        self.data_shard_cols = 100
        self.window_size = 10
        self.num_neg_sample = 30
        self.PMI_win_size = 10
        self.use_PMI = False
        self.recumpute_cooccur = True
        #self.recumpute_cooccur = False

FLAGS = Config()

help_msg = '-r <length of each sentence/line> -p <PMI window size> -w <co-occured window size> -n <number of negative samples for each mention>' \
            ' -l <max number of sentences/lines> -i <input corpus path> -c <output intermediate file path> -o <output file path>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "hp:w:r:l:i:c:o:n:")
except getopt.GetoptError:
    print help_msg
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print help_msg
        sys.exit()
    elif opt in ("-r"):
        FLAGS.data_shard_cols = int(arg)
    elif opt in ("-p"):
        FLAGS.use_PMI = True
        FLAGS.PMI_win_size = int(arg)
    elif opt in ("-w"):
        FLAGS.window_size = int(arg)
    elif opt in ("-n"):
        FLAGS.num_neg_sample = int(arg)
    elif opt in ("-l"):
        FLAGS.data_shard_rows = int(arg)
    elif opt in ("-i"): 
        FLAGS.training_file = arg
    elif opt in ("-c"):
        FLAGS.output_coocur_file = arg
    elif opt in ("-o"): 
        FLAGS.output_training_file = arg

print("Building dataset")
data, count, dictionary, reverse_dictionary = utils.build_dataset(FLAGS.training_file, FLAGS.vocab_size, FLAGS.data_shard_rows * FLAGS.data_shard_cols)
FLAGS.vocab_size = len(count)
print("vocab size: ", FLAGS.vocab_size)
data_shared = np.array(data).reshape((FLAGS.data_shard_rows, FLAGS.data_shard_cols))




if FLAGS.recumpute_cooccur:
    print("Calculating co-occurrances")
    cooccur_mat = {}
    for i in xrange(FLAGS.vocab_size):
        cooccur_mat[i] = {}
    for i in xrange(FLAGS.data_shard_rows):
        for px in xrange(FLAGS.data_shard_cols - 1):
            end_ind = min(px + FLAGS.PMI_win_size , FLAGS.data_shard_cols)
            for py in range(px + 1, end_ind):
                x = data_shared[i, px]
                y = data_shared[i, py]
                if y not in cooccur_mat[x]: cooccur_mat[x][y] = 0
                if x not in cooccur_mat[y]: cooccur_mat[y][x] = 0
                cooccur_mat[x][y] += 1
                cooccur_mat[y][x] += 1
    with open(FLAGS.output_coocur_file,'w') as f_out:
        json.dump(cooccur_mat,f_out,indent = 1)
else:
    print("loading cooccur matrix")
    with open(FLAGS.output_coocur_file) as f_in:
        cooccur_mat_load = json.load(f_in)
    cooccur_mat = {}
    print("changing format of cooccur matrix")
    for x_str in cooccur_mat_load:
        cooccur_mat[int(x_str)] = {}
        for y_str in cooccur_mat_load[x_str]:
            cooccur_mat[int(x_str)][int(y_str)] = cooccur_mat_load[x_str][y_str]

print("Calculating PMI filters")
weights = []
num_filtered_out = 0
filter_out_arr = np.zeros(FLAGS.vocab_size)
for i in xrange(FLAGS.data_shard_rows):
    row_weight = []
    for px, x in enumerate(data_shared[i, :]):
        slice_cnt = 0
        row_weight.append([])
        for idy, py in enumerate(range(px - FLAGS.window_size, px) + range(px + 1, px + FLAGS.window_size + 1)):
            if py < 0 or py >= FLAGS.data_shard_cols:
                row_weight[px].append(0.0)
                continue
            y = data_shared[i, py]

            if FLAGS.use_PMI and x in cooccur_mat and y in cooccur_mat[x]:
                if FLAGS.data_shard_rows * FLAGS.data_shard_cols * cooccur_mat[x][y] < count[x][1] * count[y][1] * FLAGS.num_neg_sample:
                    num_filtered_out += 1
                    filter_out_arr[x] += 1
                    row_weight[px].append(0.0)
                    continue

            row_weight[px].append(1.0)
            slice_cnt += 1
    weights.append(row_weight)
    print("\rFinished %6d / %6d." % (i, FLAGS.data_shard_rows)),
    sys.stdout.flush()

print("\n# filtered out: %d" % num_filtered_out)
for i in range(5):
    filter_rate = filter_out_arr[i]/float(count[i][1]*2*FLAGS.window_size)
    print(str(count[i])+' '+ str(filter_rate))
open(FLAGS.output_training_file, 'wb').write(msgpack.packb((data, count, dictionary, reverse_dictionary, weights)))
