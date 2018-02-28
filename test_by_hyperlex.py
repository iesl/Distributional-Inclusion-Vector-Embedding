import time
from gensim.models import word2vec
from eval_utils import *
from scipy import stats
import getopt
import sys

use_hist_emb=True

average_filtering = False

ent_path = ''
reverse_all_value = False

POS_delimiter = ''
include_oov = False

cross_validation_fold = 0

help_msg = 'python test_by_hyperlex.py -e <evaluation_file> -i <histogram_embedding_raw> -s <sparse_BoW> -w <gensim_model> -o <scoring_output> -r (reverse embedding) -g (global filter using the average embedding) -l [pre-computed entropy file]' \
           ' -a [The delimiter between each word and its POS) -u (include oov) -f (compute F1 and accuracy by cross-validation) -m [Gaussian mixture embedding]'
try:
    opts, args = getopt.getopt(sys.argv[1:], "he:i:s:w:o:rgl:a:uf:m:")
except getopt.GetoptError:
    print help_msg
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print help_msg
        sys.exit()
    elif opt in ("-e"):
        eval_path = arg
    elif opt in ("-i"): #s and i are mutually exclusive
        dense_BoW_path = arg
        use_hist_emb = True
    elif opt in ("-s"): #s and i are mutually exclusive
        sparse_BoW_path = arg
        use_hist_emb = False
    elif opt in ("-w"):
        word2vec_path = arg
    elif opt in ("-o"):
        out_path = arg
    elif opt in ("-r"):
        reverse_all_value = True
    elif opt in ("-l"):
        ent_path = arg
    elif opt in ("-a"):
        POS_delimiter = arg
    elif opt in ("-u"):
        include_oov = True
    elif opt in ("-f"):
        cross_validation_fold = int(arg)
    elif opt in ("-g"):
        average_filtering = True



def is_hyper_rel(rel_str):
    if rel_str[:3]=='hyp':
        return True
    else:
        return False


def is_hyper_rel_1(rel_str):
    if (rel_str == 'hyp-1'):
        return True
    else:
        return False


#WORD1 WORD2 POS TYPE AVG_SCORE AVG_SCORE_0_10 STD SCORES..
#mascara lipstick N cohyp 0.71 1.18 1.33 0 1 2 0 1 5 1 0 0 0 0 0 0 0
#highway road N hyp-1 5.80 9.67 0.40 5 6 6 6 6 6 6 6 6 5
def load_eval_file(f_in):
    eval_data = []
    child_parent_d2_info = {}
    line_count = 0
    for line in f_in:
        line_count += 1
        if line_count == 1:
            continue
        line = line[:-1]
        line_split = line.split(' ')
        #print line_split
        child_raw, parent_raw, pos, rel, score = line_split[:5]
        child = child_raw.lower()
        parent = parent_raw.lower()
        if len(POS_delimiter) > 0:
            pos = pos.lower()
            child = child + '|' + pos
            parent = parent + '|' + pos

        eval_data.append([child, parent, pos, rel, float(score)])
    return eval_data


def compute_performance(result_list_method):
    prec_list = []
    correct_count = 0
    all_count = 0
    rel_d2_count = {}
    rel_d2_prec_list = {}
    for i in range(len(result_list_method)):
        all_count += 1
        rel=result_list_method[i][2][1]
        if is_hyper_rel(rel):
            correct_count += 1
            prec_list.append(correct_count / float(all_count))
        else:
            if rel not in rel_d2_count:
                rel_d2_count[rel]=0
                rel_d2_prec_list[rel]=[]
            rel_d2_count[rel]+=1
            rel_d2_prec_list[rel].append(correct_count / float(correct_count+rel_d2_count[rel]))
    return np.mean(prec_list),rel_d2_prec_list,rel_d2_count



print "loading eval file"
with open(eval_path) as f_in:
    eval_data = load_eval_file(f_in)

print "loading word2vec file"
word2vec_model = word2vec.Word2Vec.load(word2vec_path)


t = time.time()

if (use_hist_emb):
    print "loading histogram emb"
    with open(dense_BoW_path) as f_in:
        neighbor_w = json.load(f_in)
        w_d2_ind = {}
else:
    print "loading sparse BoW"
    with open(sparse_BoW_path) as f_in:
        neighbor_w, w_d2_ind, ind_l2_w = json.load(f_in)

if reverse_all_value == True:
    print "reversing embedding"
    neighbor_w = reverse_emb(neighbor_w)

if average_filtering:
    print "filter all embedding using the global average embedding"
    neighbor_w = average_test_filtering(neighbor_w, eval_data)

if len(ent_path) > 0:
    print "loading entropy"
    with open(ent_path) as f_in:
        w_ind_d2_ent = load_entropy_file(f_in)
else:
    w_ind_d2_ent = []

elapsed = time.time() - t
print "total spent time", elapsed

parent_d2_sup_child = {}
result_list, spec_correct, method_list, method_ind_map, total_hyper_num, oov_num = compute_method_scores(eval_data, w_d2_ind, neighbor_w, is_hyper_rel, word2vec_model.wv, w_ind_d2_ent, parent_d2_sup_child, include_oov)

for method in spec_correct:
    print method, ", direction correctness: ", np.mean(spec_correct[method]), ", num support: ", len(spec_correct[method])

result_list_T = zip(*result_list)
info_list = zip(*result_list_T[2])
hyper_score = info_list[-1]

rel_list = ['hyp-1','hyp-2','hyp-3','hyp-4', 'syn', 'ant', 'cohyp', 'mero', 'no-rel', 'r-hyp-1', 'r-hyp-2', 'r-hyp-3', 'r-hyp-4']

method_and_rank_coeff = []
method_and_linear_coeff = []
for method in method_list:
    if(len(parent_d2_sup_child)==0 and method in ["num_training_child","max_training_sim","max_sim_times_num_training"]):
        continue
    m_ind = method_ind_map[method]
    result_list_method = sorted(result_list, key=lambda x: x[m_ind])
    output_AP(result_list_method, is_hyper_rel, method, rel_list, cross_validation_fold)
    sp_score, sp_p = stats.spearmanr(result_list_T[m_ind],hyper_score)
    p_score, p_p = stats.pearsonr(result_list_T[m_ind], hyper_score)
    method_and_rank_coeff.append([method, sp_score])
    method_and_linear_coeff.append([method, p_score])
    print "Spearman rank score: ", sp_score, " Pearson rank score: ", p_score

method_and_rank_coeff = sorted(method_and_rank_coeff, key=lambda x: x[1])
print "The top3 overall rank coefficient: ", method_and_rank_coeff[:3]
method_and_linear_coeff = sorted(method_and_linear_coeff, key=lambda x: x[1])
print "The top3 overall linear coefficient: ", method_and_linear_coeff[:3]

method_d2_rank = {m: v for m,v in method_and_rank_coeff}
method_d2_linear = {m: v for m,v in method_and_linear_coeff}

method_list_print = ['rnd','word2vec','AL1', 'CDE', 'entropy_diff', 'summation','summation_dot_product']


print "AP, F, accuracy, direciton"
for method in method_list_print:
    print method, ', ', -method_d2_rank[method], ', ', -method_d2_linear[method]

print 'max scores, ', -method_and_rank_coeff[0][1], ', ', -method_and_linear_coeff[0][1]
print 'which, ', method_and_rank_coeff[0][0], ', ', method_and_linear_coeff[0][0]

output_method_list = ["invCL", "CDE", "AL1", "AL1_small_w", "entropy_diff", "word2vec", "word2vec_entropy", "rnd",
"AL1_word2vec_entropy", "AL1_word2vec", "ent_dot_product", "Weed_word2vec_ent",
"summation", "summation_word2vec", "dot_product", "summation_dot_product", "CDE_summation_word2vec",
"CDE_ent", "CDE_ent_word2vec","CDE_summation_dot_product","2_norm","2_norm_word2vec","2_norm_dot_product"]


f_out = open(out_path, 'w')

f_out.write( eval_path + ', ' + str(len(result_list)) + ', , ' + str(oov_num) + '\n')

for method in output_method_list:
    f_out.write(method + ', ' + str(-method_d2_rank[method]) + ', ' + str(-method_d2_linear[method]) )
    f_out.write('\n')

f_out.write('\n')

f_out.close()

