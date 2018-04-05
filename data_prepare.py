import argparse
import cPickle as pickle
import json
import os
import random
import re
from collections import defaultdict
from tqdm import tqdm
import nltk

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='../data/eclipse')
parser.add_argument('-r', '--ratio', type=float, default=0.9)
parser.add_argument('-wv', '--word_vocab', type=int, default=20000)
parser.add_argument('-cv', '--char_vocab', type=int, default=100)
args = parser.parse_args()

UNK = 1


def read_pairs():
  bug_pairs = []
  bug_ids = set()
  with open(os.path.join(args.data, 'pairs.json'), 'r') as f:
    count = 0
    for line in f:
      count += 1
      if count > 10000:
        break

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


def func_name_tokenize(text):
  s = []
  for i, c in enumerate(text):
    if c.isupper() and i > 0 and text[i-1].islower():
      s.append(' ')
    s.append(c)
  return ''.join(s).strip()


def normalize_text(text):
  try:
    tokens = re.compile(r'[\W_]+', re.UNICODE).split(text)
    text = ' '.join([func_name_tokenize(token) for token in tokens])
    text = re.sub(r'\d+((\s\d+)+)?', 'number', text)
    return ' '.join([word.lower() for word in nltk.word_tokenize(text)])
  except:
    return 'description'


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


def normalized_data(bug_ids):
  products = set()
  bug_severities = set()
  priorities = set()
  versions = set()
  components = set()
  bug_statuses = set()
  text = []
  normalized_bugs = open(os.path.join(args.data, 'normalized_bugs.json'), 'w')
  with open(os.path.join(args.data, 'bugs.json'), 'r') as f:
    count = 0
    loop = tqdm(f)
    for line in loop:
      bug = json.loads(line)
      bug_id = int(bug['bug_id'])
      if bug_id not in bug_ids:
        continue

      count += 1
      loop.set_postfix(count=count)

      products.add(bug['product'])
      bug_severities.add(bug['bug_severity'])
      priorities.add(bug['priority'])
      versions.add(bug['version'])
      components.add(bug['component'])
      bug_statuses.add(bug['bug_status'])
      bug['description'] = normalize_text(bug['description'])
      if 'short_desc' in bug:
        bug['short_desc'] = normalize_text(bug['short_desc'])
      else:
        bug['short_desc'] = ''
      bug.pop('_id', None)
      bug.pop('delta_ts', None)
      bug.pop('creation_ts', None)
      normalized_bugs.write('{}\n'.format(json.dumps(bug)))

      text.append(bug['description'])
      text.append(bug['short_desc'])
  save_dict(products, 'product.dic')
  save_dict(bug_severities, 'bug_severity.dic')
  save_dict(priorities, 'priority.dic')
  save_dict(versions, 'version.dic')
  save_dict(components, 'component.dic')
  save_dict(bug_statuses, 'bug_status.dic')
  return text


def data_split(bug_pairs):
  random.shuffle(bug_pairs)
  split_idx = int(len(bug_pairs) * args.ratio)
  with open(os.path.join(args.data, 'train.txt'), 'w') as f:
    for pair in bug_pairs[:split_idx]:
      f.write("%d %d\n" % pair)
  test_data = {}
  for pair in bug_pairs[split_idx:]:
    bug1 = int(pair[0])
    bug2 = int(pair[1])
    if bug1 not in test_data:
      test_data[bug1] = set()
    test_data[bug1].add(bug2)
  with open(os.path.join(args.data, 'test.txt'), 'w') as f:
    for bug in test_data.keys():
      f.write("{} {}\n".format(bug, ' '.join([str(x) for x in test_data[bug]])))


def build_freq_dict(train_text):
  print('building frequency dictionaries')
  word_freq = defaultdict(int)
  char_freq = defaultdict(int)
  for text in tqdm(train_text):
    for word in text.split():
      word_freq[word] += 1
    for char in text:
      char_freq[char] += 1
  return word_freq, char_freq


def save_vocab(freq_dict, vocab_size, filename):
  top_tokens = sorted(freq_dict.items(), key=lambda x: -x[1])[:vocab_size - 2]
  print('most common token is %s which appears %d times' % (top_tokens[0][0], top_tokens[0][1]))
  print('less common token is %s which appears %d times' % (top_tokens[-1][0], top_tokens[-1][1]))
  vocab = {}
  i = 2  # 0-index is for padding, 1-index is for UNKNOWN
  for j in range(len(top_tokens)):
    vocab[top_tokens[j][0]] = i
    i += 1
  with open(os.path.join(args.data, filename), 'wb') as f:
    pickle.dump(vocab, f)
  return vocab


def build_vocabulary(train_text):
  word_freq, char_freq = build_freq_dict(train_text)
  print('word vocabulary')
  word_vocab = save_vocab(word_freq, args.word_vocab, 'word_vocab.pkl')
  print('character vocabulary')
  char_vocab = save_vocab(char_freq, args.char_vocab, 'char_vocab.pkl')
  return word_vocab, char_vocab


def dump_bugs(word_vocab, char_vocab):
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
    loop = tqdm(f)
    for line in loop:
      loop.set_description('Data dumping')
      bug = json.loads(line)
      bug['product'] = product_dict[bug['product']]
      bug['bug_severity'] = bug_severity_dict[bug['bug_severity']]
      bug['priority'] = priority_dict[bug['priority']]
      bug['version'] = version_dict[bug['version']]
      bug['component'] = component_dict[bug['component']]
      bug['bug_status'] = bug_status_dict[bug['bug_status']]
      bug['description_word'] = [word_vocab.get(w, UNK) for w in bug['description'].split()]
      bug['description_char'] = [char_vocab.get(c, UNK) for c in bug['description']]
      if len(bug['short_desc']) == 0:
        bug['short_desc'] = bug['description'][:10]
      bug['short_desc_word'] = [word_vocab.get(w, UNK) for w in bug['short_desc'].split()]
      bug['short_desc_char'] = [char_vocab.get(c, UNK) for c in bug['short_desc']]
      bug.pop('description')
      bug.pop('short_desc')
      with open(os.path.join(bug_dir, bug['bug_id'] + '.pkl'), 'wb') as f:
        pickle.dump(bug, f)


def main():
  bug_pairs, bug_ids = read_pairs()
  print("Number of bugs: {}".format(len(bug_ids)))
  print("Number of pairs: {}".format(len(bug_pairs)))

  data_split(bug_pairs)
  text = normalized_data(bug_ids)

  word_vocab, char_vocab = build_vocabulary(text)
  dump_bugs(word_vocab, char_vocab)


if __name__ == '__main__':
  main()
