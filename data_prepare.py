import argparse
import json
import os
import pickle
import random
import re
from collections import defaultdict

import nltk
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='../data/eclipse')
parser.add_argument('-r', '--ratio', type=float, default=0.8)
parser.add_argument('-v', '--vocab', type=int, default=50000)
args = parser.parse_args()

UNK = 2


def read_pairs():
  bug_pairs = []
  bug_ids = set()
  with open(os.path.join(args.data, 'pairs.json'), 'r') as f:
    for line in f:
      pair = json.loads(line)
      bug_pairs.append((int(pair['bug1']), int(pair['bug2'])))
      bug_ids.add(int(pair['bug1']))
      bug_ids.add(int(pair['bug2']))
  with open(os.path.join(args.data, 'bug_pairs.txt'), 'w') as f:
    for pair in bug_pairs:
      f.write("%d %d\n" % pair)
  bug_ids = sorted(bug_ids)
  with open(os.path.join(args.data, 'bug_ids.txt'), 'w') as f:
    for bug_id in bug_ids:
      f.write("%d\n" % bug_id)
  return bug_pairs, bug_ids


def normalize_text(text):
  text = re.sub(r"!+", "!", text)
  text = re.sub(r"\.+", ".", text)
  text = re.sub(r"\\n", " ", text)
  text = re.sub(r"\\t", " ", text)
  return ' '.join(nltk.word_tokenize(text))


def save_dict(set, filename):
  with open(os.path.join(args.data, filename), 'w') as f:
    for i, item in enumerate(set):
      f.write('%s\t%d\n' % (item, i))


def load_dict(filename):
  dict = {}
  with open(os.path.join(args.data, filename), 'r') as f:
    for line in f:
      tokens = line.split('\t')
      dict[tokens[0]] = tokens[1]
  return dict


def normalized_data(bug_ids, train_bug_ids):
  products = set()
  bug_severities = set()
  priorities = set()
  versions = set()
  components = set()
  bug_statuses = set()
  train_text = []
  normalized_bugs = open(os.path.join(args.data, 'normalized_bugs.json'), 'w')
  with open(os.path.join(args.data, 'bugs.json'), 'r') as f:
    count = 0
    for line in tqdm(f):
      # count += 1
      # if count == 500:
      #   break

      bug = json.loads(line)
      bug_id = int(bug["bug_id"])
      if bug_id not in bug_ids:
        continue

      products.add(bug['product'])
      bug_severities.add(bug['bug_severity'])
      priorities.add(bug['priority'])
      versions.add(bug['version'])
      components.add(bug['component'])
      bug_statuses.add(bug['bug_status'])

      bug.pop('delta_ts', None)
      bug.pop('creation_ts', None)

      bug['description'] = normalize_text(bug['description'])
      if 'short_description' in bug:
        bug['short_description'] = normalize_text(bug['short_description'])
      else:
        bug['short_description'] = ''
      normalized_bugs.write('{}\n'.format(json.dumps(bug)))
      if bug_id in train_bug_ids:
        train_text.append(bug['description'])
        train_text.append(bug['short_description'])
  save_dict(products, 'product.dic')
  save_dict(bug_severities, 'bug_severity.dic')
  save_dict(priorities, 'priority.dic')
  save_dict(versions, 'version.dic')
  save_dict(components, 'component.dic')
  save_dict(bug_statuses, 'bug_status.dic')
  return train_text


def data_spit(bug_pairs):
  random.shuffle(bug_pairs)
  split_idx = int(len(bug_pairs) * args.ratio)
  train_bug_ids = set()
  with open(os.path.join(args.data, 'train.txt'), 'w') as f:
    for pair in bug_pairs[:split_idx]:
      f.write("%d %d\n" % pair)
      train_bug_ids.add(int(pair[0]))
      train_bug_ids.add(int(pair[1]))
  with open(os.path.join(args.data, 'test.txt'), 'w') as f:
    for pair in bug_pairs[split_idx:]:
      f.write("%d %d\n" % pair)
  return train_bug_ids


def build_word_freq(train_text):
  word_freq = defaultdict(int)
  for text in tqdm(train_text):
    for word in text.split():
      word_freq[word] += 1
  return word_freq


def build_vocabulary(train_text):
  print('building vocabulary...')
  word_freq = build_word_freq(train_text)
  top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:args.vocab - 2]
  print('most common word is %s which appears %d times' % (top_words[0][0], top_words[0][1]))
  print('less common word is %s which appears %d times' % (top_words[-1][0], top_words[-1][1]))
  vocab = {}
  i = 2  # 0-index is for padding, 1-index is for UNKNOWN word
  for word, freq in top_words:
    vocab[word] = i
    i += 1
  with open(os.path.join(args.data, 'vocab.pkl'), 'wb') as f:
    pickle.dump(vocab, f)
  return vocab


def dump_bugs(vocab):
  bug_dir = os.path.join(args.data, 'bugs')
  if not os.path.exists(bug_dir):
    os.mkdir(bug_dir)
  product_dict = load_dict('product.dic')
  bug_severity_dict = load_dict('bug_severity.dic')
  priority_dict = load_dict('priority.dic')
  version_dict = load_dict('version.dic')
  component_dict = load_dict('component.dic')
  bug_status_dict = load_dict('bug_status.dic')
  with open(os.path.join(args.data, 'normalized_bugs.json'), 'r') as f:
    for line in f:
      bug = json.loads(line)
      bug['product'] = product_dict[bug['product']]
      bug['bug_severity'] = bug_severity_dict[bug['bug_severity']]
      bug['priority'] = priority_dict[bug['priority']]
      bug['version'] = version_dict[bug['version']]
      bug['component'] = component_dict[bug['component']]
      bug['bug_status'] = bug_status_dict[bug['bug_status']]
      bug['description'] = [vocab.get(w, UNK) for w in bug['description'].split()]
      bug['short_description'] = [vocab.get(w, UNK) for w in bug['short_description'].split()]
      with open(os.path.join(bug_dir, bug['bug_id'] + '.pkl'), 'wb') as f:
        pickle.dump(bug, f)


def main():
  bug_pairs, bug_ids = read_pairs()
  print("Number of bugs: {}".format(len(bug_ids)))
  print("Number of pairs: {}".format(len(bug_pairs)))

  train_bug_ids = data_spit(bug_pairs)
  train_text = normalized_data(bug_ids, train_bug_ids)
  print("Number of train docs: {}".format(len(train_text)))

  vocab = build_vocabulary(train_text)
  dump_bugs(vocab)


if __name__ == '__main__':
  main()


