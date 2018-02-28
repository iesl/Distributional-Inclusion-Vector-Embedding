import gzip
import getopt
import sys

with_stopwords = True
have_POS = False

stop_word_file_name = './stop_word_list'

row_length = 100

help_msg = '-p <delimiter for appended POS of each token> -s <file path storing stop words> -r <length of each sentence/line>' \
            ' -l <max number of sentences/lines> -i <input corpus path> -o <output file path>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "hp:s:r:l:i:o:")
except getopt.GetoptError:
    print help_msg
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print help_msg
        sys.exit()
    elif opt in ("-p"):
        POS_delimiter = arg
        have_POS = True
    elif opt in ("-s"):
        stop_word_file_name = arg
        with_stopwords = False
    elif opt in ("-r"):
        row_length = int(arg)
    elif opt in ("-l"):
        max_line = int(arg)
    elif opt in ("-i"): 
        input_file_name = arg
    elif opt in ("-o"): 
        output_file_name = arg



if input_file_name[-3:] == '.gz':
    f = gzip.open(input_file_name)
else:
    f = open(input_file_name)
line = f.readline()

if output_file_name[-3:] == '.gz':
    out = gzip.open(output_file_name,'w')
else:
    out = open(output_file_name,'w')
    
if not with_stopwords:
    stopwords = set(open(stop_word_file_name).read().splitlines())

wordcnt = 0
current_row = []
current_set = set()

line_count = 0
while len(line) > 0:
    #tokens = line.lower().strip(delimiter).split()
    tokens = line.lower().split()
    for x in tokens:
        wordcnt += 1
        if have_POS:
            last_delimiter = x.rfind(POS_delimiter)
            x_org = x[:last_delimiter]
            if not x_org.isalpha(): 
                wordcnt -= 1
                continue
            if not with_stopwords and x_org in stopwords: continue
            current_row.append(x[:last_delimiter+2])
        else:
            if not x.isalpha(): 
                wordcnt -= 1
                continue
            if not with_stopwords and x in stopwords: continue
            current_row.append(x)
            #if x not in current_set:
            #    current_row.append(x)
            #    current_set.add(x)
        if len(current_row) == row_length:
            out.write(' '.join(current_row) + '\n')
            current_row = []
            current_set = set()
            line_count+=1
        if wordcnt >= max_line*row_length:
            break
    #if line_count>=max_line:
    if wordcnt >= max_line*row_length:
        break
    line = f.readline()

print wordcnt
print line_count
f.close()
out.close()
