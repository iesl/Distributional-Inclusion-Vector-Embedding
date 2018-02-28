from gensim.models import word2vec
import gzip
import time

input_data_path='./data/wackypedia_512k_matrix_wostop'
#input_data_path='./data/wackypedia_POS_matrix_wostop'

#stop_word_path='/iesl/canvas/hschang/code/taxonomy/resources/stop_word_list'
stop_word_path='/dev/null'

gensim_out_path='./model/gensim_model_lower'
#gensim_out_path='./model/gensim_model_lower_POS'

emb_dim=100
max_sent_num=512000

#add_POS = True
add_POS = False

def load_stop_words(f_in):
    stop_w_list = set()
    for line in f_in:
        stop_w_list.add(line[:-1])
    return stop_w_list

def norm_w(w):
    return w.lower()

def check_is_stop(w_n, stop_w_list):
    global add_POS
    if not add_POS:
        return (w_n in stop_w_list)
    else:
        last_ind = w_n.rfind('|')
        if last_ind < 0:
            print "assume not having POS"
            add_POS = False
            return (w_n in stop_w_list)
        return (w_n[:last_ind] in stop_w_list)

print("loading stop words")
with open(stop_word_path) as f_in:
    stop_w_list = load_stop_words(f_in)


data = []
print "loading file"
if input_data_path[-3:] == '.gz':
    f_in = gzip.open(input_data_path)
else:
    f_in = open(input_data_path)
i=0
for line in f_in:
    line=line[:-1]
    if(i>=max_sent_num):
        break
    line_norm=[]
    for w in line.split():
        w_n = norm_w(w)
        if( check_is_stop(w_n, stop_w_list) ):
            continue
        line_norm.append(w_n)
    i+=1
    data.append(line_norm)
print data[:2]
f_in.close()
t = time.time()

print "training word2vec"
model = word2vec.Word2Vec(data, size=emb_dim, window=10, min_count=10, workers=8, iter=5)

model.save(gensim_out_path)

elapsed = time.time() - t
print "total spent time", elapsed
