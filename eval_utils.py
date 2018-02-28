import numpy as np
import random
import json
import math
from scipy import spatial
import sys

def reverse_emb(neighbor_w):
    max_value = 0
    all_ind = set()
    for w in neighbor_w:
        if(len(neighbor_w[w]) == 0):
            continue
        max_value_w = np.max(neighbor_w[w].values())
        max_value = max(max_value, max_value_w)
        inds = neighbor_w[w].keys()
        all_ind = all_ind.union(set(inds[:]))
    
    neighbor_w_r = {}
    for w in neighbor_w:
        neighbor_w_r[w] = {}
        for emb_ind in all_ind:
            if emb_ind not in neighbor_w[w]:
                neighbor_w_r[w][emb_ind] = max_value
                continue
            emb_value = neighbor_w[w][emb_ind]
            if(max_value == emb_value):
                continue
            else:
                neighbor_w_r[w][emb_ind] = max_value - emb_value
    
    return neighbor_w_r


def one_side_loss(hist_parent_s, hist_child_s):
    loss_parent = 0
    total_count_parent = np.sum(hist_parent_s.values())
    for ind in hist_parent_s:
        hist_parent_i = hist_parent_s[ind]
        if (ind not in hist_child_s):
            hist_child_i = 0
            continue
        hist_child_i = hist_child_s[ind]
        hist_diff_i = hist_parent_i - hist_child_i
        if (hist_diff_i > 0):
            loss_parent += hist_diff_i
    if(total_count_parent>0):
        out = loss_parent / float(total_count_parent)
    else:
        out = 1
    return out


def AL1_loss_raw(hist_parent_s, hist_child_s, over_penalty):
    loss_parent_norm = one_side_loss(hist_parent_s, hist_child_s)
    loss_child_norm = one_side_loss(hist_child_s, hist_parent_s)
    loss = over_penalty * loss_child_norm + loss_parent_norm
    return loss


def compute_loss(prob_parent_s, prob_child_s, sol_a, over_penalty):
    loss = 0
    for ind in prob_parent_s:
        prob_parent_i = prob_parent_s[ind]
        if (ind not in prob_child_s):
            loss += prob_parent_i
            continue
        prob_child_i = prob_child_s[ind]
        prob_diff_i = prob_parent_i - prob_child_i * sol_a
        if (prob_diff_i > 0):
            loss += prob_diff_i
        else:
            loss -= over_penalty * prob_diff_i
    return loss


def solve_LP_by_sort(prob_child_s, prob_parent_s, over_penalty):
    poss_a_prob = []
    poss_a_0 = []
    poss_a_0_sum = 0
    target_value = 1 / float(over_penalty + 1)
    for ind in prob_child_s:
        if prob_child_s[ind] == 0:
            continue
        if (ind not in prob_parent_s):
            poss_a_0.append((0, prob_child_s[ind]))
            poss_a_0_sum += prob_child_s[ind]
        else:
            poss_a = prob_parent_s[ind] / prob_child_s[ind]
            poss_a_prob.append((poss_a, prob_child_s[ind]))
    if (poss_a_0_sum >= target_value or len(poss_a_prob) == 0):
        return 1, 0
    target_value_remain = target_value - poss_a_0_sum
    poss_a_prob_sorted = sorted(poss_a_prob, key=lambda x: x[0])

    prob_sum = 0
    for j in range(len(poss_a_prob_sorted)):
        over_index = j
        prob_sum += poss_a_prob_sorted[j][1]
        if (prob_sum >= target_value_remain):
            break
    sol_a = poss_a_prob_sorted[over_index][0]
    loss = compute_loss(prob_parent_s, prob_child_s, sol_a, over_penalty)

    if (over_index > 0):
        sol_a_2 = poss_a_prob_sorted[over_index - 1][0]
        loss_2 = compute_loss(prob_parent_s, prob_child_s, sol_a_2, over_penalty)
        if (loss_2 < loss):
            loss = loss_2
            sol_a = sol_a_2
    return loss, sol_a


def compute_ent(prob_list):
    prob_np = np.array(prob_list)
    ent = -np.sum(prob_np * np.log(prob_np))
    return ent


def compute_prob_ent(sBoW_dict):
    total_count = float(sum(sBoW_dict.values()))
    prob_dict = {}
    
    for ind in sBoW_dict:
        prob_i = sBoW_dict[ind] / total_count
        prob_dict[ind] = prob_i
    

    ent = compute_ent(prob_dict.values())

    return ent, prob_dict


def get_ent_prob(child_ind, w_d2_cache, neighbor_w):
    if (child_ind in w_d2_cache):
        ent_child, prob_child, child_nei = w_d2_cache[child_ind]
    else:
        child_nei = neighbor_w[child_ind]
        ent_child, prob_child = compute_prob_ent(child_nei)
        w_d2_cache[child_ind] = [ent_child, prob_child, child_nei]
    return ent_child, prob_child, child_nei


def compute_CDE(prob_child, prob_parent):
    inter_sum = 0
    norm_sum = 0
    for w in prob_child:
        norm_sum += prob_child[w]
        if (w in prob_parent):
            inter_sum += min(prob_child[w], prob_parent[w])
    if(norm_sum > 0 ):
        CDE_score = inter_sum / float(norm_sum)
    else:
        CDE_score = 0

    return CDE_score


def compute_Weed(hist_child, hist_parent):
    inter_sum = 0
    norm_sum = 0
    for w in hist_child:
        norm_sum += hist_child[w]
        if w in hist_parent:
            inter_sum += hist_child[w]
    if norm_sum > 0:
        Weed_score = inter_sum / float(norm_sum)
    else:
        Weed_score = 0

    return Weed_score


def load_entropy_file(f_in):
    w_ind_and_ent = json.load(f_in)
    w_ind_d2_ent = {w_ind: v for w_ind, v in w_ind_and_ent}
    return w_ind_d2_ent


def compute_nei_ent(w_ind_d2_ent,child_nei,top_neighbor_k):
    child_nei_sorted = sorted(child_nei.items(), key=lambda x:x[1], reverse=True)
    ent_list=[]
    for i in range( min(top_neighbor_k, len(child_nei_sorted)) ):
        w_ind, freq = child_nei_sorted[i]
        ent_list.append(w_ind_d2_ent[w_ind])
    E_child = np.median(ent_list)
    return E_child


def compute_SLQS_sub(w_ind_d2_ent, child_nei, parent_nei):
    top_neighbor_k=100
    E_child = compute_nei_ent(w_ind_d2_ent, child_nei, top_neighbor_k)
    E_parent = compute_nei_ent(w_ind_d2_ent, parent_nei, top_neighbor_k)
    SLQS_sub = E_parent - E_child
    return SLQS_sub


def compute_norm_s(emb_1_s):
    emb_1_v = np.array(emb_1_s.values())
    return np.sqrt( np.sum(emb_1_v*emb_1_v) )


def compute_dot_product(emb_1_s, emb_2_s):
    if len(emb_1_s) < len(emb_2_s):
        emb_less_s = emb_1_s
        emb_more_s = emb_2_s
    else:
        emb_less_s = emb_2_s
        emb_more_s = emb_1_s

    dot_prod = 0
    for w in emb_less_s:
        if w in emb_more_s:
            dot_prod += emb_less_s[w] * emb_more_s[w]

    return dot_prod


def compute_cosine_sim(emb_1_s,emb_2_s):
    emb_12_dot = compute_dot_product(emb_1_s, emb_2_s)
    emb_12_norm_dot = compute_norm_s(emb_1_s) * compute_norm_s(emb_2_s)
    if(emb_12_norm_dot > 0):
        return emb_12_dot/emb_12_norm_dot
    else:
        return 0


def sparse_to_dense(w_sparse_rep, bin_num):
    w_desne_rep = np.zeros(bin_num)
    w_desne_rep[map(int, w_sparse_rep.keys())] = w_sparse_rep.values()
    return w_desne_rep


def dense_to_sparse(w_dense_rep):
    w_sparse_rep = {}
    #print np.nonzero(w_dense_rep)
    for nei_topic in np.nonzero(w_dense_rep)[0].tolist():
        w_sparse_rep[nei_topic] = w_dense_rep[nei_topic]
    return w_sparse_rep



def compute_L1_dist(prob_child, prob_parent):
    parent_remain = 1
    L1_prob = 0
    for w in prob_child:
        if w not in prob_parent:
            L1_prob += prob_child[w]
        else:
            L1_prob += abs(prob_child[w]-prob_parent[w])
            parent_remain -= prob_parent[w]

    L1_prob += parent_remain
    return L1_prob


def compute_coverage_dist(hist_child, hist_parent):
    penalty = 0
    for w in hist_child:
        if w not in hist_parent:
            penalty += hist_child[w]
        else:
            penalty += max(hist_child[w]-hist_parent[w],0)

    return penalty


def sparse_normalization(prob_inter):
    total_prob = np.sum(prob_inter.values())
    if total_prob <= 0:
        return prob_inter

    prob_inter_norm = {emb_ind: emb_value/total_prob for emb_ind, emb_value in prob_inter.items()}
    return prob_inter_norm


def combine_by_prod(prob_1,prob_2,prod_smooth_const):
    prob_inter = {emb_ind: math.sqrt(emb_value*prod_smooth_const) for emb_ind, emb_value in prob_1.items()}

    for emb_ind, emb_value in prob_2.items():
        if emb_ind not in prob_inter:
            prob_inter[emb_ind] = math.sqrt(emb_value*prod_smooth_const)
        else:
            hist_emb_1_value = prob_inter[emb_ind]*prob_inter[emb_ind]/prod_smooth_const
            prob_inter[emb_ind] = math.sqrt(hist_emb_1_value*emb_value)

    return sparse_normalization(prob_inter)


def combine_by_prod_hard(prob_1,prob_2,take_sqrt):
    prob_inter={}
    for emb_ind, emb_value in prob_1.items():
        if emb_ind in prob_2:
            if take_sqrt:
                prob_inter[emb_ind] = math.sqrt(emb_value*prob_2[emb_ind])
            else:
                prob_inter[emb_ind] = emb_value * prob_2[emb_ind]
    return prob_inter


def combine_by_min(prob_1,prob_2):
    prob_inter = {}
    for emb_ind, emb_value in prob_2.items():
        if emb_ind in prob_1:
            prob_inter[emb_ind] = min(prob_1[emb_ind],prob_2[emb_ind])
    
    return sparse_normalization(prob_inter)


def accumulate_hist_emb(w, neighbor_w, emb_sum, emb_count):
    if w not in neighbor_w:
        return
    for nei_w_ind, v in neighbor_w[w].items():
        if nei_w_ind not in emb_sum:
            emb_sum[nei_w_ind] = 0
        emb_sum[nei_w_ind] += v
    emb_count[0]+=1


def average_test_filtering(neighbor_w, eval_data):
    emb_sum = {}
    emb_count = [0]
    eval_set = set()
    for data_i in eval_data:
        child = data_i[0]
        parent = data_i[1]
        eval_set.add(child)
        eval_set.add(parent)
        accumulate_hist_emb(child, neighbor_w, emb_sum, emb_count)
        accumulate_hist_emb(parent, neighbor_w, emb_sum, emb_count)

    emb_avg = {}
    for w in emb_sum:
        emb_avg[w] = emb_sum[w]/emb_count[0]

    neighbor_w_new = {}
    take_sqrt = True
    for w in eval_set:
        if w not in neighbor_w:
            continue
        neighbor_w_new[w] = combine_by_prod_hard(neighbor_w[w], emb_avg, take_sqrt)

    return neighbor_w_new


def word2vec_similarity(child, parent, word2vec_wv):
    if isinstance(word2vec_wv,dict):
        return 1 - spatial.distance.cosine(word2vec_wv[child], word2vec_wv[parent])
    else:
        return word2vec_wv.similarity(child, parent)


def combine_by_accumulate(dict_1, dict_2, scale):
    for ind, v in dict_2.items():
        if ind in dict_1:
            dict_1[ind] = dict_1[ind] + v*scale
            #dict_1[ind] = max( dict_1[ind], v*scale)
        else:
            dict_1[ind] = v*scale


def compute_sparse_average(child_sparse_prob_list, scale_v):
    emb_weighted_avg = {}
    #for i, child_s_prob in enumerate(child_sparse_prob_list):
    for i in range(len(child_sparse_prob_list)):
        child_s_prob = child_sparse_prob_list[i]
        combine_by_accumulate(emb_weighted_avg, child_s_prob, scale_v[i])
    return emb_weighted_avg


def compute_sparse_diff(child_sparse_prob_list, emb_weighted_avg):
    diff_list = []
    for child_s_prob in child_sparse_prob_list:
        child_s_diff = child_s_prob.copy()
        combine_by_accumulate(child_s_diff, emb_weighted_avg, -1)
        diff_list.append(child_s_diff)
    return diff_list


def compute_sq_sum(diff_list):
    d_2_arr = []
    for diff in diff_list:
        d_2_arr.append( np.sum( np.square(diff.values()) ) )
    return np.array(d_2_arr)


def robust_estimation(child_sparse_emb_list):
    converge_err = 0.0001
    err = 1
    i = 0
    child_sparse_prob_list = []
    for child_s_emb in child_sparse_emb_list:
        total_val = float( sum(child_s_emb.values()) )
        child_s_prob = {w: v/total_val for w, v in child_s_emb.items()}
        child_sparse_prob_list.append(child_s_prob)

    scale_v = [1 / float(len(child_sparse_prob_list))] * len(child_sparse_prob_list)
    emb_weighted_avg = compute_sparse_average(child_sparse_prob_list, scale_v)

    while err > converge_err:
        diff_list = compute_sparse_diff(child_sparse_prob_list, emb_weighted_avg)
        d_2 = compute_sq_sum(diff_list)
        ro_2 = 1.4826*np.median(d_2)
        if ro_2 == 0:
            w = np.ones(d_2.shape[0])
        else:
            w = ro_2 / ((ro_2+d_2)*(ro_2+d_2))

        emb_weighted_avg_old = emb_weighted_avg
        w = w / np.sum(w)
        emb_weighted_avg = compute_sparse_average(child_sparse_prob_list, w)
        combine_by_accumulate(emb_weighted_avg_old, emb_weighted_avg, -1)

        err = np.sum( np.square(emb_weighted_avg_old.values()) )
        i = i+1
    w = w / np.mean(w)
    emb_weighted_avg = compute_sparse_average(child_sparse_emb_list, w)
    return w, emb_weighted_avg, i


def compute_method_scores(eval_data, w_d2_ind, neighbor_w, is_hyper_rel, word2vec_wv, w_ind_d2_ent, parent_d2_sup_child, include_oov):
    over_penalty_1 = 5
    over_penalty_2 = 20
    oov_list = []
    result_list = []
    w_d2_cache = {}
    spec_correct = {"AL1_diff": [], "AL1_diff_s_w": [], "AL1_raw_diff": [], "entropy_diff": [], "CDE_diff": [],
                    "SLQS_sub": [], "summation_diff": [], "2_norm_diff": []}

    method_list = ["invCL", "CDE", "CDE norm", "entropy_AL1", "AL1_diff", "AL1",
                   "AL1_diff_small_w", "AL1_small_w", "entropy_diff", "AL1_raw",
                   "AL1_raw_smaller_w", "word2vec", "word2vec_entropy", "invOrder", "rnd",
                   "CDE_diff", "SLQS_sub", "AL1_word2vec_entropy", "AL1_word2vec", "Weed", "Weed_word2vec_ent",
                   "summation", "summation_word2vec", "dot_product", "summation_dot_product", "CDE_summation_word2vec",
                   "CDE_ent", "CDE_ent_word2vec", "L1_entropy","order_org","num_training_child","max_training_sim","max_sim_times_num_training",
                   "2_norm","2_norm_word2vec","2_norm_dot_product","ent_dot_product", "CDE_summation_dot_product"]

    method_ind_map = {m: 3 + i for i, m in enumerate(method_list)}

    total_hyper_num = 0

    print "computing distances"
    for i, data_i in enumerate(eval_data):
        child = data_i[0]
        parent = data_i[1]
        rel = data_i[3]

        if len(w_d2_ind)==0:
            child_ind = child
            parent_ind = parent
        else:
            if (child not in w_d2_ind) or (parent not in w_d2_ind):
                oov_list.append(data_i)
                if include_oov:
                    result_list.append( [child, parent, data_i[2:]] + [100000000000000*(1+random.random()) ]*len(method_list) )
                    if is_hyper_rel(rel):
                        total_hyper_num += 1
                continue
            child_ind = str(w_d2_ind[child])
            parent_ind = str(w_d2_ind[parent])

        if (child_ind not in neighbor_w) or (parent_ind not in neighbor_w):
            oov_list.append(data_i)
            if include_oov:
                result_list.append([child, parent, data_i[2:]] + [100000000000000*(1+random.random())] * len(method_list))
                if is_hyper_rel(rel):
                    total_hyper_num += 1
            continue

        rnd_baseline = random.random()
        if len(w_ind_d2_ent)>0:
            SLQS_sub = compute_SLQS_sub(w_ind_d2_ent, neighbor_w[child_ind], neighbor_w[parent_ind])
        else:
            SLQS_sub = -rnd_baseline

        ent_child, prob_child, hist_child = get_ent_prob(child_ind, w_d2_cache, neighbor_w)
        ent_parent, prob_parent, hist_parent = get_ent_prob(parent_ind, w_d2_cache, neighbor_w)

        L1_prob = compute_L1_dist(prob_child, prob_parent)

        CDE_score_norm = compute_CDE(prob_child, prob_parent)
        CDE_score = compute_CDE(hist_child, hist_parent)
        CDE_score_inv = compute_CDE(hist_parent, hist_child)

        Weed_score = compute_Weed(hist_child, hist_parent)

        loss_raw_1 = AL1_loss_raw(hist_parent, hist_child, over_penalty_2)
        loss_raw_inv_1 = AL1_loss_raw(hist_child, hist_parent, over_penalty_2)
        loss_raw_2 = AL1_loss_raw(hist_parent, hist_child, over_penalty_1)

        if (child in word2vec_wv) and (parent in word2vec_wv):
            word2vec_sim = word2vec_similarity(child, parent, word2vec_wv)
        else:
            word2vec_sim = 0

        loss_1, sol_a_1 = solve_LP_by_sort(prob_child, prob_parent, over_penalty_2)
        loss_inv, sol_a_inv = solve_LP_by_sort(prob_parent, prob_child, over_penalty_2)
        loss_2, sol_a_2 = solve_LP_by_sort(prob_child, prob_parent, over_penalty_1)
        loss_inv_2, sol_a_inv_2 = solve_LP_by_sort(prob_parent, prob_child, over_penalty_1)

        num_training_child = 0
        max_sim_training_child = -1

        if parent in parent_d2_sup_child:
            num_training_child = len(parent_d2_sup_child[parent])
            if child in word2vec_wv:
                for child_training in parent_d2_sup_child[parent]:
                    if child_training in word2vec_wv:
                        max_sim_training_child = max(max_sim_training_child, word2vec_similarity(child, child_training, word2vec_wv))

        summation_parent = np.sum(hist_parent.values())
        summation_child = np.sum(hist_child.values())
        two_norm_parent = np.linalg.norm(hist_parent.values())
        two_norm_child = np.linalg.norm(hist_child.values())
        order_penalty = compute_coverage_dist(hist_child, hist_parent)

        dot_product = compute_cosine_sim(hist_parent, hist_child)

        if is_hyper_rel(rel):
            total_hyper_num += 1
            if(CDE_score_inv<CDE_score):
                spec_correct['CDE_diff'].append(1)
            elif CDE_score_inv>CDE_score:
                spec_correct['CDE_diff'].append(0)

            if (ent_child < ent_parent):
                spec_correct['entropy_diff'].append(1)
            elif (ent_child > ent_parent):
                spec_correct['entropy_diff'].append(0)

            if (loss_1 < loss_inv):
                spec_correct['AL1_diff'].append(1)
            elif (loss_1 > loss_inv):
                spec_correct['AL1_diff'].append(0)

            if (loss_2 < loss_inv_2):
                spec_correct['AL1_diff_s_w'].append(1)
            elif (loss_2 > loss_inv_2):
                spec_correct['AL1_diff_s_w'].append(0)

            if (loss_raw_1 < loss_raw_inv_1):
                spec_correct['AL1_raw_diff'].append(1)
            elif (loss_raw_1 > loss_raw_inv_1):
                spec_correct['AL1_raw_diff'].append(0)

            if (SLQS_sub>0):
                spec_correct['SLQS_sub'].append(1)
            elif(SLQS_sub<0):
                spec_correct['SLQS_sub'].append(0)

            if (summation_parent > summation_child):
                spec_correct['summation_diff'].append(1)
            elif(summation_parent < summation_child):
                spec_correct['summation_diff'].append(0)
            
            if (two_norm_parent > two_norm_child):
                spec_correct['2_norm_diff'].append(1)
            elif(two_norm_parent < two_norm_child):
                spec_correct['2_norm_diff'].append(0)

        result_list.append([child, parent, data_i[2:], -(1 - CDE_score_inv) * CDE_score, -CDE_score, -CDE_score_norm,
                            (1.01 - loss_1) * (ent_child - ent_parent), loss_1 - loss_inv, loss_1, loss_2 - loss_inv_2,
                            loss_2, ent_child - ent_parent, loss_raw_1, loss_raw_2, -word2vec_sim,
                            word2vec_sim * (ent_child - ent_parent), loss_1 * (1.1 - loss_inv), rnd_baseline,
                            CDE_score_inv - CDE_score, -SLQS_sub, word2vec_sim * (ent_child - ent_parent)*(1.1 - loss_1), (loss_1 - 1.1)*word2vec_sim, Weed_score, (ent_child - ent_parent) * Weed_score * word2vec_sim,
                            summation_child-summation_parent, (summation_child-summation_parent)*word2vec_sim, -dot_product, (summation_child-summation_parent)*dot_product,  (summation_child-summation_parent) * CDE_score * word2vec_sim,
                            (ent_child - ent_parent) * CDE_score, (ent_child - ent_parent) * CDE_score * word2vec_sim, (ent_child - ent_parent)*(1-L1_prob), order_penalty, -num_training_child, -max_sim_training_child, -max_sim_training_child*num_training_child,
                            (two_norm_child-two_norm_parent), (two_norm_child-two_norm_parent)*word2vec_sim, (two_norm_child-two_norm_parent)*dot_product, (ent_child - ent_parent)*dot_product,CDE_score*(summation_child-summation_parent)*dot_product ])

    print "total eval count", len(result_list)
    print "total oov count", len(oov_list)
    print oov_list[: min(len(oov_list), 10)]

    return result_list, spec_correct, method_list, method_ind_map, total_hyper_num, len(oov_list)


def compute_F1_accuracy_by_cross_val(cross_validation_fold, result_list_method, is_hyper_rel, num_hyper):
    total_sample_num = len(result_list_method)
    random_ind = np.random.permutation(np.arange(total_sample_num) % cross_validation_fold)
    sample_num_testing = np.zeros(cross_validation_fold)
    hyper_num_testing = np.zeros(cross_validation_fold)
    for i in range(total_sample_num):
        current_ind = random_ind[i]
        sample_num_testing[current_ind] += 1
        rel = result_list_method[i][2][1]
        if is_hyper_rel(rel):
            hyper_num_testing[current_ind] += 1

    hyper_num_training = num_hyper - hyper_num_testing
    sample_num_training = total_sample_num - sample_num_testing

    for j in range(cross_validation_fold):
        if hyper_num_testing[j] == 0:
            print "too few data or too many cross validation splits"
            sys.exit()

    total_num_sum = 0
    total_num_testing = np.zeros(cross_validation_fold)
    correct_num_sum = 0
    correct_num_testing = np.zeros(cross_validation_fold)
    best_training_F1 = np.zeros(cross_validation_fold)
    best_testing_F1 = np.zeros(cross_validation_fold)
    best_training_accuracy = np.zeros(cross_validation_fold)
    best_testing_accuracy = np.zeros(cross_validation_fold)
    for i in range(total_sample_num):
        current_ind = random_ind[i]
        total_num_testing[current_ind] += 1
        rel = result_list_method[i][2][1]
        total_num_sum += 1
        if is_hyper_rel(rel):
            correct_num_sum += 1
            correct_num_testing[current_ind] += 1
            correct_num_training = correct_num_sum - correct_num_testing
            total_num_training = total_num_sum - total_num_testing
            for j in range(cross_validation_fold):
                if total_num_training[j] == 0 or correct_num_training[j] ==0:
                    continue
                precision_j = float(correct_num_training[j]) / total_num_training[j]
                recall_j = float(correct_num_training[j]) / hyper_num_training[j]
                F1_j = 2*precision_j*recall_j/(recall_j+precision_j)
                if F1_j > best_training_F1[j]:
                    if total_num_testing[j] == 0 or correct_num_testing[j]==0:
                        continue
                    best_training_F1[j] = F1_j
                    precision_j = float(correct_num_testing[j]) / total_num_testing[j]
                    recall_j = float(correct_num_testing[j]) / hyper_num_testing[j]
                    F1_testing_j = 2 * precision_j * recall_j / (recall_j + precision_j)
                    best_testing_F1[j] = F1_testing_j
                remaining_hyper = hyper_num_training[j] - correct_num_training[j]
                remaining_rest = sample_num_training[j] - total_num_training[j] - remaining_hyper
                accuracy_j = float(correct_num_training[j] + remaining_rest) / sample_num_training[j]
                if accuracy_j > best_training_accuracy[j]:
                    best_training_accuracy[j] = accuracy_j
                    remaining_hyper = hyper_num_testing[j] - correct_num_testing[j]
                    remaining_rest = sample_num_testing[j] - total_num_testing[j] - remaining_hyper
                    best_testing_accuracy[j] = float(correct_num_testing[j]+remaining_rest) / sample_num_testing[j]

    F1_sum = 0
    accuracy_sum = 0
    for j in range(cross_validation_fold):
        F1_sum += best_testing_F1[j]*sample_num_testing[j]
        accuracy_sum += best_testing_accuracy[j]*sample_num_testing[j]

    return F1_sum / total_sample_num, accuracy_sum / total_sample_num


def compute_AP(result_list_method, is_hyper_rel, rel_list, cross_validation_fold):
    prec_list = []
    correct_count = 0
    all_count = 0
    rel_d2_count = {}
    rel_d2_prec_list = {}
    for rel in rel_list:
        #if rel not in rel_d2_count:
        rel_d2_count[rel] = 0
        rel_d2_prec_list[rel] = []

    total_sample_num = len(result_list_method)
    num_hyper = 0
    for i in range(total_sample_num):
        rel = result_list_method[i][2][1]
        if is_hyper_rel(rel):
            num_hyper += 1

    if cross_validation_fold > 0:
        avg_F1, avg_accuracy = compute_F1_accuracy_by_cross_val(cross_validation_fold, result_list_method, is_hyper_rel, num_hyper)

    best_accuracy = 0
    best_F1 = 0
    for i in range(total_sample_num):
        all_count += 1
        rel = result_list_method[i][2][1]
        if is_hyper_rel(rel):
            correct_count += 1
            precision = correct_count / float(all_count)
            prec_list.append(precision)
            remaining_hyper = num_hyper - correct_count
            remaining_rest = total_sample_num - all_count - remaining_hyper
            total_correct_count = correct_count + remaining_rest
            accuracy = total_correct_count/float(total_sample_num)
            recall = correct_count / float(num_hyper)
            F1 = 2*precision*recall/(recall+precision)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            if F1 > best_F1:
                best_F1 = F1

            for rel in rel_d2_count:
                rel_d2_prec_list[rel].append(correct_count / float(correct_count+rel_d2_count[rel]))
        else:
            rel_d2_count[rel] += 1

    for rel in rel_list:
        if rel_d2_count[rel] == 0:
            rel_d2_count.pop(rel, None)
            rel_d2_prec_list.pop(rel, None)

    if cross_validation_fold > 0:
        return np.mean(prec_list), rel_d2_prec_list, rel_d2_count, avg_accuracy, avg_F1
    else:
        return np.mean(prec_list), rel_d2_prec_list, rel_d2_count, best_accuracy, best_F1

def MAP_by_average(e_d2_prec_list, e_d2_all_count):
    MAP_weighted_sum = 0
    MAP_weights = 0
    for entity in e_d2_prec_list:
        if len(e_d2_prec_list[entity]) == 0:
            continue
        weight = e_d2_all_count[entity]
        AP = np.mean(e_d2_prec_list[entity])
        MAP_weighted_sum += weight*AP
        MAP_weights += weight
    MAP_overall = MAP_weighted_sum / MAP_weights
    return MAP_overall


def compute_MAP_given_entity(result_list_method, is_hyper_rel, entity_ind):
    e_d2_prec_list = {}
    e_d2_correct_count = {}
    e_d2_all_count = {}

    for i in range(len(result_list_method)):
        rel = result_list_method[i][2][1]
        entity = result_list_method[i][entity_ind]
        if entity not in e_d2_prec_list:
            e_d2_prec_list[entity] = []
            e_d2_correct_count[entity] = 0
            e_d2_all_count[entity] = 0

        e_d2_all_count[entity] += 1

        if is_hyper_rel(rel):
            e_d2_correct_count[entity] += 1
            e_d2_prec_list[entity].append(e_d2_correct_count[entity] / float(e_d2_all_count[entity]))
        
    MAP_overall = MAP_by_average(e_d2_prec_list, e_d2_all_count)
    
    return MAP_overall


def output_AP(result_list_method, is_hyper_rel, method, rel_list, cross_validation_fold):
    AP, rel_d2_prec_list, rel_d2_count, best_accuracy, best_F1 = compute_AP(result_list_method, is_hyper_rel, rel_list, cross_validation_fold)
    
    print method, ": overall AP, ", "%.1f"%(AP*100), ", best accuracy, ", "%.1f"%(best_accuracy*100), ", best F1, ", "%.1f"%(best_F1*100), ", ", np.sum(rel_d2_count.values()), "; ",
    for rel in rel_list:
        if rel in rel_d2_prec_list:
            print rel, ", ", "%.1f" % (np.mean(rel_d2_prec_list[rel])*100), ", ", str(rel_d2_count[rel]), "; ",
    MAP_child = compute_MAP_given_entity(result_list_method, is_hyper_rel, 0)
    MAP_parent = compute_MAP_given_entity(result_list_method, is_hyper_rel, 1)
    print method, ":child overall, ", "%.1f" % (MAP_child * 100), ", ",
    print method, ":parent overall, ", "%.1f" % (MAP_parent * 100), "; ",
    print

    return AP, MAP_child, MAP_parent, best_accuracy, best_F1


def accuracy_by_guess_unknown(spec_correct_m, total_hyper_num):
    number_of_guessing = total_hyper_num - len(spec_correct_m)
    number_of_guess_correct = number_of_guessing // 2
    return float(np.sum(spec_correct_m)+number_of_guess_correct) / total_hyper_num
