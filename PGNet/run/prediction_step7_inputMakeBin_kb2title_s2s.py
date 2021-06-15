# -*- coding: UTF-8 -*-
import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
from collections import defaultdict
import jieba

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

domain = sys.argv[2] #"chuju"
dataset = sys.argv[3] #'test'
tableFile = sys.argv[4] #table or table.comp
exp = domain + '_' + dataset + '_title_bin'
#data_dir = 'data'
#data_dir = '/export/homes/baojunwei/alpha-w-pipeline/data'
data_dir = os.path.join(sys.argv[1], 'data')

test_data_files = os.path.join(data_dir, domain + '_' + dataset, dataset + "." + tableFile)
test_label_files = os.path.join(data_dir, domain + '_' + dataset, dataset + ".fakeWriting")

print(test_data_files)
print(test_label_files)
finished_files_dir = os.path.join(data_dir, exp, "bin")
print(finished_files_dir)
chunks_dir = os.path.join(data_dir, exp, "chunked")

VOCAB_SIZE = 100000
CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data

"""
将finished_files 分割成1000个chunk
"""


def chunk_file(set_name):
  in_file = os.path.join(finished_files_dir, '%s.bin' % set_name)
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk))  # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['test']:
    print("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name)
  print("Saved chunked data in %s" % chunks_dir)


def tokenize_stories(stories_dir, tokenized_stories_dir):
  """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
  print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
  stories = os.listdir(stories_dir)
  # make IO list file
  print("Making list of files to tokenize...")
  with open("mapping.txt", "w") as f:
    for s in stories:
      f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))
  command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
  print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
  subprocess.call(command)
  print("Stanford CoreNLP Tokenizer has finished.")
  os.remove("mapping.txt")

  # Check that the tokenized stories directory contains the same number of files as the original directory
  num_orig = len(os.listdir(stories_dir))
  num_tokenized = len(os.listdir(tokenized_stories_dir))
  if num_orig != num_tokenized:
    raise Exception(
      "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
      tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
  print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def read_writing_text_file(text_file, encoder_type):
  index = 0
  lines = defaultdict(list)
  records = []
  with open(text_file, "r", encoding=encoder_type) as f:
    for line in f:
      text = line.strip()
      label_lines[label_index].append(' '.join(jieba.cut(text)))
      label_index += 1

  for index, items in lines.items():
    abstract = SENTENCE_START + " " + items[0] + " " + SENTENCE_END
    label_records.append(abstract)
  return records


def read_label_kb_file(label_file, kb_file, encoder_type):
  label_lines = defaultdict(list)
  label_records = []
  table_lines = defaultdict(list)
  table_records = []
  table_index = 0
  label_index = 0


  with open(kb_file, "r", encoding=encoder_type) as f:
    for line in f:
      kvs = line.strip().split(' ## ')
      for kv in kvs:
        parts = kv.split(' $$ ')
        key = parts[0].strip()
        value = parts[1].strip()
        table_lines[table_index].append(key + " $$ " + ' '.join(jieba.cut(value)))
      table_index += 1

  with open(label_file, "r", encoding=encoder_type) as f:
    for line in f:
      text = line.strip()
      label_lines[label_index].append(' '.join(jieba.cut(text)))
      label_index += 1

  for index, items in label_lines.items():
    abstract = SENTENCE_START + " " + items[0] + " " + SENTENCE_END
    label_records.append(abstract)
    table_item = ' ## '.join(table_lines[index])
    table_records.append(table_item)


  return label_records, table_records


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s.encode())
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line == "": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_ques_ans(data_file, label_file, encoder_type):
  label_lines, data_lines = read_label_kb_file(label_file, data_file, encoder_type)

  # Lowercase everything
  data_lines = [line.lower() for line in data_lines]
  label_lines = [line.lower() for line in label_lines]

  return data_lines, label_lines


def write_to_bin(data_file, label_file, out_file, encoder_type, makevocab=False):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""

  if makevocab:
    vocab_counter = collections.Counter()

  questions, answers = get_ques_ans(data_file, label_file, encoder_type)
  num_q = len(questions)
  num_a = len(answers)
  print(num_q)
  print(num_a)
  num_k = min(num_q, num_a)

  with open(out_file, 'wb') as writer:
    for idx in range(num_k):
      # Get the strings to write to .bin file
      """
      核心二进制代码
      """
      question = questions[idx]
      answer = answers[idx]
      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([question.encode()])
      tf_example.features.feature['abstract'].bytes_list.value.extend([answer.encode()])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      # Write the vocab to file, if applicable
      if makevocab:
        ques_tokens = question.split(' ')
        ans_tokens = answer.split(' ')
        ans_tokens = [t for t in ans_tokens if t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
        tokens = ques_tokens + ans_tokens
        tokens = [t.strip() for t in tokens]  # strip
        tokens = [t for t in tokens if t != ""]  # remove empty
        vocab_counter.update(tokens)

  print("Finished writing file %s\n" % out_file)

  # write vocab to file
  if makevocab:
    print("Writing vocab file...")
    with open(os.path.join(finished_files_dir, "vocab"), 'w', encoding=encoder_type) as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print("Finished writing vocab file")


def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception(
      "stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


if __name__ == '__main__':
  # Create some new directories
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  encode_type = 'utf-8'
  # Read the tokenized stories, do a little postprocessing then write to bin files
  write_to_bin(test_data_files, test_label_files, os.path.join(finished_files_dir, "test.bin"), encode_type)

  # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
  chunk_all()
  
