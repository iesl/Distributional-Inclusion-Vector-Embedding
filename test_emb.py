import json
import numpy as np
import time
from gensim.models import word2vec
import random
import getopt
import sys
import math
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from scipy import sparse
from eval_utils import *


use_hist_emb=True
#use_hist_emb = False

average_filtering = False

#eval_verb = True
eval_verb = False

ent_path = ''
reverse_all_value = False

POS_delimiter = ''

training_path = ''
training_sample_num = 0
compositional_mode = ''
include_oov = False
cross_validation_fold = 0
visualize_metric_name = ''

help_msg = 'python test_emb.py -e <evaluation_file> -i <histogram_embedding_raw> -s <sparse_BoW> -w <gensim_model> -l [pre-computed entropy file] -a [The delimiter between each word and its POS) ' \
           '-o <scoring_output> -r (reverse embedding) -v (including verbs in BLESS) -g (global filter using the average embedding) -u (include oov) -f (compute F1 and accuracy by cross-validation)' \
           '-t [the path of training data] -n [the number of training samples for each parent] -c [es:sp:as] (testing entities are compositional. es: entities sum, as: all sum, sp: sentence product) -z [visualized metric name]'
try:
    opts, args = getopt.getopt(sys.argv[1:], "he:i:s:w:o:rvgt:n:c:l:a:uf:z:")
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
    elif opt in ("-v"):
        eval_verb = True
    elif opt in ("-g"):
        average_filtering = True
    elif opt in ("-t"):
        training_path = arg
    elif opt in ("-l"):
        ent_path = arg
    elif opt in ("-a"):
        POS_delimiter = arg
    elif opt in ("-u"):
        include_oov = True
    elif opt in ("-f"):
        cross_validation_fold = int(arg)
    elif opt in ("-z"):
        visualize_metric_name = arg
    elif opt in ("-n"):
        try:
            training_sample_num = int(arg)
        except ValueError:
            training_sample_num = arg
    elif opt in ("-c"):
        if arg != 'es' and arg != 'sp' and arg != 'as':
            print "-c [es:sp:as]"
            sys.exit()
        compositional_mode = arg





def is_hyper_rel(rel_str):
    if rel_str=='hyper':
        return True
    else:
        return False


# goldfish-n      demon-n         False   random-n
# herring-n       fish-n  True    hyper
def load_eval_file(f_in):
    eval_data = []
    child_parent_d2_info = {}
    eval_child_set = set()
    eval_all_set = set()
    for line in f_in:
        line = line[:-1]
        # print line
        child_pos, parent_pos, is_hyper, rel = line.split('\t')
        child_pos = child_pos.strip()
        parent_pos = parent_pos.strip()
        is_hyper = is_hyper.strip()
        rel = rel.strip()
        if child_pos[-2]=='-':
            if len(POS_delimiter) == 0:
                child = child_pos[:-2].lower()
                parent = parent_pos[:-2].lower()
            else:
                child = child_pos.lower().replace('-',POS_delimiter)
                parent = parent_pos.lower().replace('-', POS_delimiter)
        else:
            child = child_pos
            parent = parent_pos
        if not eval_verb and child_pos[-2:] != parent_pos[-2:]:
            continue
        eval_data.append([child, parent, is_hyper, rel])
        eval_child_set.add(child)
        eval_all_set.add(child)
        eval_all_set.add(parent)

    random.shuffle(eval_data)
    return eval_data, eval_child_set, eval_all_set


def load_training_for_classifier(f_in):
    training_data = []
    for line in f_in:
        line = line[:-1]
        child_pos, parent_pos, is_hyper, rel = line.split('\t')
        if child_pos[-2] == '-':
            child = child_pos[:-2].lower()
            parent = parent_pos[:-2].lower()
        else:
            child = child_pos
            parent = parent_pos
        if (not eval_verb) and (child_pos[-2:] != parent_pos[-2:]):
            continue
        training_data.append( [child, parent, is_hyper_rel(rel) ] )

    return training_data


def convert_training_to_feature_info(training_data, neighbor_w, word2vec_wv):
    row_ind = 0
    max_ind = 0
    feature_info_list = []
    for i in range(len(training_data)):
        child, parent, gt = training_data[i]
        if (child not in neighbor_w) or (parent not in neighbor_w) or (child not in word2vec_wv) or (parent not in word2vec_wv):
            continue
        child_parent_diff = neighbor_w[parent].copy()
        combine_by_accumulate(child_parent_diff, neighbor_w[child], -1)
        parent_max_ind = 0        
        child_max_ind = 0        
        if len(neighbor_w[parent])>0:
            parent_max_ind = np.amax(map(int,neighbor_w[parent].keys()))
        if len(neighbor_w[child])>0:
            child_max_ind = np.amax(map(int,neighbor_w[child].keys()))
        max_ind = np.amax( [max_ind, parent_max_ind, child_max_ind])
        word2vec_sim = word2vec_similarity(child, parent, word2vec_wv)
        feature_info_list.append( [row_ind, neighbor_w[parent], neighbor_w[child], child_parent_diff, word2vec_sim, gt] )

        row_ind += 1
    return max_ind, feature_info_list


def convert_feature_info_to_sparse_matrix(feature_info_list, max_ind_all):
    row_ind_all = []
    col_ind_all = []
    data_ind_all = []
    y = []
    for row_ind, parent_dict, child_dict, diff_dict, word2vec_sim, gt in feature_info_list:
        col_ind_i = map(int, diff_dict.keys()) + [max_ind_all+1,max_ind_all+2,max_ind_all+3,max_ind_all+4]
        col_ind_all += col_ind_i
        row_ind_all += [row_ind]*len(col_ind_i)
        data_ind_all += diff_dict.values() + [np.sum(parent_dict.values()), np.sum(child_dict.values()), compute_cosine_sim(parent_dict,child_dict), word2vec_sim ]
        y.append(gt)

    X = sparse.csr_matrix( (data_ind_all, (row_ind_all, col_ind_all)), shape = (len(feature_info_list), max_ind_all+5) )
    print X.shape
    return X, y



def load_training(f_in, eval_child_set):
    p_d2_c_training = {}
    for line in f_in:
        line = line[:-1]
        child_pos, parent_pos, is_hyper, rel = line.split('\t')
        if not is_hyper_rel(rel):
            continue
        if child_pos[-2] == '-':
            child = child_pos[:-2].lower()
            parent = parent_pos[:-2].lower()
        else:
            child = child_pos
            parent = parent_pos
        if (not eval_verb) and (child_pos[-2:] != parent_pos[-2:]):
            continue
        if child in eval_child_set:
            continue
        if parent not in p_d2_c_training:
            p_d2_c_training[parent] = []
        p_d2_c_training[parent].append( child )

    return p_d2_c_training


def simple_supervision(neighbor_w, training_sample_num, p_d2_c_training, w_d2_ind):
    discard_org_emb = False
    if training_sample_num < 0:
        training_sample_num = abs(training_sample_num)
        discard_org_emb = True

    parent_d2_sup_child = {}
    total_supervision_num = 0
    total_parent_num = 0
    for parent in p_d2_c_training:
        if len(w_d2_ind) > 0:
            if parent not in w_d2_ind:
                continue
            parent_ind = str(w_d2_ind[parent])
        else:
            parent_ind = parent
        if parent_ind not in neighbor_w:
            continue
        if parent not in parent_d2_sup_child:
            if discard_org_emb:
                neighbor_w[parent_ind] = {}
            parent_d2_sup_child[parent] = []

        random.shuffle(p_d2_c_training[parent])
        num_sup = 0
        child_sparse_emb_list = []
        for child in p_d2_c_training[parent]:
            if len(w_d2_ind) > 0:
                if child not in w_d2_ind:
                    continue
                child_ind = str(w_d2_ind[child])
            else:
                child_ind = child
            if child_ind not in neighbor_w:
                continue
            combine_by_accumulate(neighbor_w[parent_ind], neighbor_w[child_ind], 1)
            
            num_sup += 1
            parent_d2_sup_child[parent].append(child)
            if num_sup >= training_sample_num:
                break
        
        total_supervision_num += num_sup
        total_parent_num += 1
        
    print "average supervision for each parent: ", total_supervision_num / float(total_parent_num)
    print "total number of supervision pairs:", str(total_supervision_num)

    return neighbor_w, parent_d2_sup_child


def compose_embedding(neighbor_w, eval_all_set, w_d2_ind, compositional_mode):
    take_sqrt = False
    for sent in eval_all_set:
        word_list = sent.split(',')
        if len(word_list) <= 1:
            w = word_list[0]
            if len(w_d2_ind) > 0 and w in w_d2_ind:
                w_ind = str(w_d2_ind[w])
                neighbor_w[w] = neighbor_w[w_ind]
            continue
        word_emb_prod = {}
        all_in_wv = True
        #for w in word_list:
        for i in range(len(word_list)):
            w = word_list[i]
            if compositional_mode == 'es' and i == 1:
                continue
            if len(w_d2_ind) > 0 and w in w_d2_ind:
                w = str(w_d2_ind[w])
            if len(POS_delimiter) > 0 and i < len(word_list)-1:
                w = w + sent[-2:]
            if w not in neighbor_w:
                all_in_wv = False
                break
            
            if compositional_mode == 'sp':
                if len(word_emb_prod) == 0:
                    word_emb_prod = neighbor_w[w]
                else:
                    combine_by_prod_hard(word_emb_prod, neighbor_w[w], take_sqrt)
            elif compositional_mode == 'es' or compositional_mode == 'as':
                for nei_w_ind, v in neighbor_w[w].items():
                    if nei_w_ind not in word_emb_prod:
                        word_emb_prod[nei_w_ind] = 0
                    word_emb_prod[nei_w_ind] += v
        if all_in_wv:
            if compositional_mode == 'sp':
                neighbor_w[sent] = word_emb_prod
            else:
                word_emb_num = float(len(word_list))
                neighbor_w[sent] = {w: v/word_emb_num for w,v in word_emb_prod.items()}
            
    return neighbor_w


def compose_word2vec(word2vec_model,eval_all_set):
    word2vec_dict = {}
    for sent in eval_all_set:
        word_list = sent.split(',')
        word_emb_sum = np.zeros(word2vec_model.vector_size)
        all_in_wv = True
        
        for i,w in enumerate(word_list):
            if len(POS_delimiter) > 0 and i < len(word_list)-1:
                w = w + sent[-2:]
            if w not in word2vec_model.wv:
                all_in_wv = False
                break
            word_emb_sum += word2vec_model.wv[w]
        if all_in_wv:
            word_emb_num = len(word_list)
            word2vec_dict[sent] = word_emb_sum/word_emb_num
    return word2vec_dict


def train_and_test_classifier(X_train, y_train, X_eval, y_eval):
    classifier = linear_model.LogisticRegression()
    classifier.fit(X_train, y_train)
    y_est = classifier.predict(X_eval)
    print "Accuracy: ", metrics.accuracy_score(y_eval, y_est)
    print "F1: ", metrics.f1_score(y_eval, y_est)
    y_prob = classifier.predict_proba(X_eval)
    print "AP: ", metrics.average_precision_score(y_eval, y_prob[:, 1])


def score_to_classifier_data(result_list, all_method_ind_list):
    num_samples = len(result_list)
    X_eval = np.zeros( (num_samples, len(all_method_ind_list)) )
    y_eval = np.zeros( num_samples )
    for i in range(num_samples):
        y_eval[i] = is_hyper_rel(result_list[i][2][1])
        for j,m_ind in enumerate(all_method_ind_list):
            X_eval[i, j] = result_list[i][m_ind]
    return X_eval, y_eval


def ensemble_supervision(result_list, result_list_training, all_method_ind_list):
    X_eval, y_eval = score_to_classifier_data(result_list, all_method_ind_list)
    X_train, y_train = score_to_classifier_data(result_list_training, all_method_ind_list)
    train_and_test_classifier(X_train, y_train, X_eval, y_eval)

def compute_average_sparsity(neighbor_w, w_d2_ind,eval_all_set):
    dim_sum = 0
    inside_vocab =0
    for w_c in eval_all_set:
        if ',' in w_c:
            w_list = w_c.split(',')
        else:
            w_list = [w_c]

        for w in w_list:
            if len(w_d2_ind)>0:
                if w not in w_d2_ind:
                    continue
                w_ind = str(w_d2_ind[w])
            else:
                w_ind = w
            if w_ind not in neighbor_w:
                continue
            inside_vocab += 1
            dim_sum += len(neighbor_w[w_ind])
         
    print float(inside_vocab) / len(eval_all_set)
    return float(dim_sum) / inside_vocab 

print "loading word2vec file"
word2vec_model = word2vec.Word2Vec.load(word2vec_path)

print "loading eval file"
with open(eval_path) as f_in:
    eval_data, eval_child_set, eval_all_set = load_eval_file(f_in)
    
t = time.time()

if use_hist_emb:
    print "loading histogram emb"
    with open(dense_BoW_path) as f_in:
        neighbor_w = json.load(f_in)
        w_d2_ind = {}
else:
    print "loading sparse BoW"
    with open(sparse_BoW_path) as f_in:
        neighbor_w, w_d2_ind, ind_l2_w = json.load(f_in)

avg_sparsity = compute_average_sparsity(neighbor_w, w_d2_ind,eval_all_set)
print "Average sparsity: ", avg_sparsity

if reverse_all_value == True:
    print "reversing embedding"
    neighbor_w = reverse_emb(neighbor_w)

if len(training_path) > 0:
    print "loading training file"
    with open(training_path) as f_in:
        training_data_eval_format, training_child_set, training_all_set = load_eval_file(f_in)
else:
    training_all_set = set()

if len(compositional_mode) > 0:
    compose_embedding(neighbor_w, eval_all_set.union(training_all_set), w_d2_ind, compositional_mode)
    word2vec_wv = compose_word2vec(word2vec_model, eval_all_set.union(training_all_set))
    w_d2_ind = {}
else:
    word2vec_wv = word2vec_model.wv

parent_d2_sup_child = {}
if len(training_path) > 0 and training_sample_num != 0:
    with open(training_path) as f_in:
        p_d2_c_training = load_training(f_in, eval_child_set)
        if isinstance( training_sample_num, int ):
            training_sample_num_int = training_sample_num
        else:
            training_sample_num_int = 100
        neighbor_w, parent_d2_sup_child = simple_supervision(neighbor_w, training_sample_num_int, p_d2_c_training, w_d2_ind)
        if training_sample_num == 'SVM':
            with open(training_path) as f_in:
                training_data = load_training_for_classifier(f_in)
                max_ind_training, feature_info_list_training = convert_training_to_feature_info(training_data, neighbor_w, word2vec_wv)
            with open(eval_path) as f_in_eval:
                training_data_eval = load_training_for_classifier(f_in_eval)
                max_ind_eval, feature_info_list_eval = convert_training_to_feature_info(training_data_eval, neighbor_w, word2vec_wv)
            max_ind_all = max(max_ind_training, max_ind_eval)+1
            X_train, y_train = convert_feature_info_to_sparse_matrix(feature_info_list_training, max_ind_all)
            X_eval, y_eval = convert_feature_info_to_sparse_matrix(feature_info_list_eval, max_ind_all)
            if max_ind_all < 1000:
                X_train = X_train.toarray()
                X_eval = X_eval.toarray()
            train_and_test_classifier(X_train, y_train, X_eval, y_eval)



if len(ent_path) > 0:
    print "loading entropy"
    with open(ent_path) as f_in:
        w_ind_d2_ent = load_entropy_file(f_in)
else:
    w_ind_d2_ent = []



if average_filtering:
    print "filter all embedding using the global average embedding"
    neighbor_w = average_test_filtering(neighbor_w, eval_data)

elapsed = time.time() - t
print "total spent time", elapsed

result_list, spec_correct, method_list, method_ind_map, total_hyper_num, oov_num = compute_method_scores(eval_data, w_d2_ind, neighbor_w, is_hyper_rel, word2vec_wv, w_ind_d2_ent,  parent_d2_sup_child, include_oov)

if training_sample_num == 'ensemble':
    print "ensemble many unsupervised scores"
    result_list_training, spec_correct_training, method_list_training, method_ind_map_training = compute_method_scores(training_data_eval_format, w_d2_ind, neighbor_w,
                                                                                       is_hyper_rel, word2vec_wv,
                                                                                       w_ind_d2_ent, parent_d2_sup_child)

    feature_method_list = ["invCL", "CDE", "entropy_AL1", "AL1","AL1_small_w", "entropy_diff", "word2vec", "word2vec_entropy",
                    "invOrder", "AL1_word2vec_entropy", "AL1_word2vec", "Weed", "Weed_word2vec_ent",
                    "summation", "summation_word2vec", "dot_product", "summation_dot_product", "CDE_summation_word2vec",
                    "CDE_ent", "CDE_ent_word2vec", "L1_entropy", "order_org"]
    all_method_ind_list = []
    for method in feature_method_list:
        all_method_ind_list.append(method_ind_map[method])
    ensemble_supervision(result_list, result_list_training, all_method_ind_list)


for method in spec_correct:
    print method, ", direction correctness: ", np.mean(spec_correct[method]), ", num support: ", len(spec_correct[method])




rel_list = ['random-n','random-v','random-j','event', 'mero', 'coord', 'attri', 'attrib', 'synonym', 'antonym', 'entailm', 'random']

method_and_AP = []
method_and_cMAP = []
method_and_pMAP = []
method_and_best_F1 = []
method_and_best_accuracy = []
AL1_records = {}
AL1_small_w_records = {}
for method in method_list:
    if(len(parent_d2_sup_child)==0 and method in ["num_training_child","max_training_sim","max_sim_times_num_training"]):
        continue
    result_list_method = sorted(result_list, key=lambda x: x[method_ind_map[method]])
    AP, cMAP, pMAP, best_accuracy, best_F1 = output_AP(result_list_method, is_hyper_rel, method, rel_list, cross_validation_fold)
    method_and_AP.append([method, AP])
    method_and_cMAP.append([method, cMAP])
    method_and_pMAP.append([method, pMAP])
    method_and_best_F1.append([method, best_F1])
    method_and_best_accuracy.append([method, best_accuracy])
    if method == 'AL1':
        AL1_records["AP"] = AP
        AL1_records["cMAP"] = cMAP
        AL1_records["pMAP"] = pMAP
    if method == 'AL1_small_w':
        AL1_small_w_records["AP"] = AP
        AL1_small_w_records["cMAP"] = cMAP
        AL1_small_w_records["pMAP"] = pMAP

method_and_AP = sorted(method_and_AP, key=lambda x: x[1], reverse=True)
print "The top5 overall AP: ", method_and_AP[:5]
method_and_best_F1 = sorted(method_and_best_F1, key=lambda x: x[1], reverse=True)
print "The top5 best F1: ", method_and_best_F1[:5]
method_and_best_accuracy = sorted(method_and_best_accuracy, key=lambda x: x[1], reverse=True)
print "The top5 best accuracy: ", method_and_best_accuracy[:5]

method_and_cMAP = sorted(method_and_cMAP, key=lambda x: x[1], reverse=True)
print "The top5 overall child MAP: ", method_and_cMAP[:5]
method_and_pMAP = sorted(method_and_pMAP, key=lambda x: x[1], reverse=True)
print "The top5 overall parent MAP: ", method_and_pMAP[:5]
print "Order: ", AL1_records, ", Order (small weights):", AL1_small_w_records

method_d2_AP = {m: v for m,v in method_and_AP}
method_d2_F1 = {m: v for m,v in method_and_best_F1}
method_d2_accuracy = {m: v for m,v in method_and_best_accuracy}

method_list_print = ['rnd', 'word2vec', 'AL1', 'CDE', 'order_org', 'entropy_diff', 'summation', 'summation_dot_product']

method_d2_direction = {'AL1': 'AL1_diff', 'CDE': 'CDE_diff', 'summation': 'summation_diff', 'entropy_diff': 'entropy_diff'}

print "AP, F, accuracy, direciton"
for method in method_list_print:
    print method, ', ', method_d2_AP[method], ', ', method_d2_F1[method], ', ', method_d2_accuracy[method],
    if method in method_d2_direction:
        dir_method = method_d2_direction[method]
        print ', ', np.mean(spec_correct[dir_method])
    else:
        print

print 'max scores, ', method_and_AP[0][1], ', ', method_and_best_F1[0][1], ', ', method_and_best_accuracy[0][1]
print 'which, ', method_and_AP[0][0], ', ', method_and_best_F1[0][0], ', ', method_and_best_accuracy[0][0]

output_method_list = ["invCL", "CDE", "AL1", "AL1_small_w", "entropy_diff", "word2vec", "word2vec_entropy", "rnd",
"AL1_word2vec_entropy", "AL1_word2vec", "ent_dot_product", "Weed_word2vec_ent",
"summation", "summation_word2vec", "dot_product", "summation_dot_product", "CDE_summation_word2vec",
"CDE_ent", "CDE_ent_word2vec","CDE_summation_dot_product","2_norm","2_norm_word2vec","2_norm_dot_product"]

f_out = open(out_path, 'w')

f_out.write( eval_path + ', ' + str(len(result_list))  + ', , ' + str(oov_num) + ', , , ' + str(total_hyper_num) + '\n')

for method in output_method_list:
    f_out.write(method + ', ' + str(method_d2_AP[method]) + ', ' + str(method_d2_F1[method]) + ', ' + str(
        method_d2_accuracy[method]))
    if method in method_d2_direction:
        dir_method = method_d2_direction[method]
        f_out.write(', '+str( accuracy_by_guess_unknown(spec_correct[dir_method],total_hyper_num) ) + ', ' + str(np.mean(spec_correct[dir_method]) ) )
    f_out.write('\n')

f_out.write('\n')


if len(visualize_metric_name) > 0:
    method_ind = method_ind_map[visualize_metric_name]
    result_list_sorted = sorted(result_list, key=lambda x: x[method_ind])
    for result_i in result_list_sorted:
        f_out.write(",".join(result_i[:2])+','+result_i[2][1]+',')
        f_out.write("\t".join(map(str, result_i)) + '\n')

f_out.close()

