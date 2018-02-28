from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
from random import shuffle, randint

import gzip

def read_data(filename, len_limit):
  if filename[-3:] == '.gz':
    lines = gzip.open(filename,'r').read().splitlines()
  else:
    lines = open(filename,'r').read().splitlines()
  words = []
  #shuffle(lines)
  for line in lines:
    parts = line.split()
    for p in parts:
        words.append(p)
        if len(words) >= len_limit: return words
  return words

def build_dataset(filename, vocabulary_size, len_limit):
  words = read_data(filename, len_limit)
  count = [['UUUNKKK', -1]]
  if vocabulary_size<0:
    count_threshold = -vocabulary_size
    count.extend(collections.Counter({k: c for k, c in collections.Counter(words).items() if c >= count_threshold}).most_common())
  else:
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  del words  
  return data, count, dictionary, reverse_dictionary

def getWordmap(textfile):
    words={}
    We = []
    f = open(textfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i=i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]]=n
        We.append(v)
    return (words, np.matrix(We))

def lookup(We,words,w):
    w = w.lower()
    if w in words:
        return We[words[w],:]
    else:
        return We[words['UUUNKKK'],:]

def lookup_with_unk(We,words,w):
    w = w.lower()
    if w in words:
        return We[words[w],:],False
    else:
        return We[words['UUUNKKK'],:],True

