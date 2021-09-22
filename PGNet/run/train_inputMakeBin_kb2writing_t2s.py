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
from transformers import BertTokenizer
import pdb

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

#output = "kb2writing_table2seq_bigVocab"
#input = "jiayongdianqi"
#output = "kb2writing_table2seq_filtered_bigVocab"
#input = "jiayongdianqi_filtered"
#output = "kb2writing_t2s"
#input = "chuju"

#data_dir = '/export/homes/baojunwei/alpha-w-pipeline/data'
data_dir = os.path.join(sys.argv[1], 'data')
input = sys.argv[2]
dataset = sys.argv[3]
tokenizer = BertTokenizer.from_pretrained(sys.argv[4])
output = input + "_" + dataset + "_writing_bin"

#train_data_files = os.path.join(data_dir, input, "train.table")
train_data_files = os.path.join(data_dir, input, "train.table.com")
# valid_data_files = os.path.join(data_dir, input, "dev.table.com")
#test_data_files = "/export/homes/wangyifan/kb_write/shouji/data/shouji/out1"
# test_data_files = os.path.join(data_dir, input, "test.table.com")
#train_label_files = os.path.join(data_dir, input, "train.text")

train_aspect_files = os.path.join(data_dir, input, "train.aspect")
# dev_aspect_files = os.path.join(data_dir, input, "dev.aspect")
# test_aspect_files = os.path.join(data_dir, input, "test.aspect")

train_focus_files = os.path.join(data_dir, input, "train.focus")

train_label_files = os.path.join(data_dir, input, "train.text")

# valid_label_files = os.path.join(data_dir, input, "dev.text")
#test_label_files = "/export/homes/wangyifan/kb_write/shouji/data/shouji/train.text"
# test_label_files = os.path.join(data_dir, input, "test.text")

finished_files_dir = os.path.join(data_dir, output, "bin")
print(finished_files_dir)
chunks_dir = os.path.join(data_dir, output, "chunked")

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
  for set_name in ['train', 'dev', 'test']:
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
      "The tokenized stoies directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
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


def read_label_kb_file(kb_fie, aspect_file, focus_file, label_file, encoder_type):
  focus_lines = defaultdict(list) # 每条样本的aspect
  focus_records = []
  aspect_lines = defaultdict(list) # 每条样本的aspect
  aspect_records = []
  label_lines = defaultdict(list)
  label_records = []
  caption_lines = defaultdict(list)
  caption_lines_sentence = defaultdict(list)  # 存储每个ocr结果，不合并
  caption_records = []
  caption_records_sentence = []
  attribute_lines = defaultdict(list)
  attribute_records = []
  value_lines = defaultdict(list)
  value_records = []
  table_index = 0
  label_index = 0
  aspect_index = 0
  focus_index = 0
  with open(kb_fie, "r", encoding=encoder_type) as f:
    for line in f:
      kvs = line.strip().split(' ## ')
      for kv in kvs:
        parts = kv.split(' $$ ')
        key = parts[0].replace(' ','').replace('\t','').strip()
        if(key.lower()=='skuname'):
          value = parts[1]
          buf_split = value.split('。')
#           buf_split2 = ' '.join(buf_split)
#           buf_split2_cut = [_.strip() for _ in jieba.cut(buf_split2.strip()) if _.strip() != '']
          for buf_sent in buf_split:
            v_tokens = list([_.strip() for _ in tokenizer.tokenize(buf_sent) if _.strip() != ''])
            valStr = ' '.join(v_tokens)
            caption_lines[table_index].append(valStr.lower())
#           caption_lines[table_index].append(' '.join(caption_lines_sentence[table_index]))
        else:
          value = parts[1]
          v_tokens = list([_.strip() for _ in tokenizer.tokenize(value) if _.strip() != ''])
          valStr = ' '.join(v_tokens)
          attrStr = ' '.join([key] * len(v_tokens))
          attribute_lines[table_index].append(attrStr)
          value_lines[table_index].append(valStr)
      table_index += 1

  with open(label_file, "r", encoding=encoder_type) as f:
    for line in f:
      text = line.strip()
      label_lines[label_index].append(' '.join(tokenizer.tokenize(text)))
      label_index += 1

  with open(focus_file, "r", encoding=encoder_type) as f:
    for line in f:
      text = line.strip()
      focus_lines[focus_index].append(text)
      focus_index += 1
    
  with open(aspect_file, "r", encoding=encoder_type) as f:
    for line in f:
      text = line.strip()
      aspect_lines[aspect_index].append(text)
      aspect_index += 1
    
  cnt = 0
  for index, items in label_lines.items():
    abstract = SENTENCE_START + " " + items[0] + " " + SENTENCE_END
    aspect_records.append(aspect_lines[index][0])
    focus_records.append(focus_lines[index][0])
    label_records.append(abstract)
    caption_records.append('$$$'.join(caption_lines[index]))
#     caption_records_sentence.append('###'.join(caption_lines_sentence[index]))
    attribute_records.append('\t'.join(attribute_lines[index]))
    value_records.append('\t'.join(value_lines[index]))
    l1 = len(attribute_lines[index])
    l2 = len(value_lines[index])
    l3 = len(' '.join(attribute_lines[index]).split(' '))
    l4 = len(' '.join(value_lines[index]).split(' '))
    
    
    if(l1 != l2 or l3 != l4):
      print(index)
      print(l1)
      print(l2)
      print(l3)
      print(l4)
      print('=============================')
      cnt += 1
  print('error %s'%cnt)

  return caption_records, attribute_records, value_records, label_records, aspect_records, focus_records

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


def get_ques_ans(kb_file, aspect_file, focus_file, label_file, encoder_type):
  caption_lines, attribute_lines, value_lines, label_lines, aspect_lines, focus_lines = \
    read_label_kb_file(kb_file, aspect_file, focus_file, label_file, encoder_type)
  # Lowercase everything
  caption_lines = [line.lower() for line in caption_lines]
  attribute_lines = [line.lower() for line in attribute_lines]
  value_lines = [line.lower() for line in value_lines]
  label_lines = [line.lower() for line in label_lines]
  aspect_lines = [line.lower() for line in aspect_lines]
  focus_lines = [line.lower() for line in focus_lines]
  return caption_lines, attribute_lines, value_lines, label_lines, aspect_lines, focus_lines


def write_to_bin(kb_file, aspect_file, focus_file, label_file, out_file, encoder_type, makevocab=False):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""

  if makevocab:
    vocab_counter = collections.Counter()
    vocab_aspect = collections.Counter()

  captions, attributes, values, writings, aspects, focuses = get_ques_ans(kb_file, aspect_file, focus_file, label_file, encoder_type)
  num_q = len(captions)
  num_attr = len(attributes)
  num_val = len(values)
  num_a = len(writings)
  print(num_q)
  print(num_attr)
  print(num_val)
  print(num_a)
  num_k = min(num_q, num_a, num_attr, num_val)

  with open(out_file, 'wb') as writer:
    for idx in range(num_k):
      # Get the strings to write to .bin file
      """
      核心二进制代码
      """
      caption = captions[idx]
      attribute = attributes[idx]
      value = values[idx]
      writing = writings[idx]
      aspect = aspects[idx]
      focus = focuses[idx]
      if (idx <= 3):
        print('caption:', caption)
        print('attribute:', attribute)
        print('value:', value)
        print('writing:', writing)
        print('aspect:', aspect)
        print('focus:', focus)
      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([caption.encode()])
      tf_example.features.feature['attribute'].bytes_list.value.extend([attribute.encode()])
      tf_example.features.feature['value'].bytes_list.value.extend([value.encode()])
      tf_example.features.feature['abstract'].bytes_list.value.extend([writing.encode()])
      tf_example.features.feature['aspect'].bytes_list.value.extend([aspect.encode()])
      tf_example.features.feature['focus'].bytes_list.value.extend([focus.encode()])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))
      # Write the vocab to file, if applicable
      if makevocab:
        ques_tokens = ' '.join(caption.split("$$$")).split()
        attribute_tokens = attribute.split()
        value_tokens = value.split()
        ans_tokens = writing.split()
        ans_tokens = [t for t in ans_tokens if t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
        tokens = ques_tokens + attribute_tokens + value_tokens + ans_tokens
        tokens = [t.strip() for t in tokens]  # strip
        tokens = [t for t in tokens if t != ""]  # remove empty
        vocab_counter.update(tokens)
        vocab_aspect.update([aspect])

  print("Finished writing file %s\n" % out_file)

  # write vocab to file
  if makevocab:
    print("Writing vocab file...")
    with open(os.path.join(finished_files_dir, "vocab"), 'w', encoding=encoder_type) as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
        
    with open(os.path.join(finished_files_dir, "vocab_aspect"), 'w', encoding=encoder_type) as writer:
      for word, count in vocab_aspect.most_common(VOCAB_SIZE):
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
  print('start writing train file...')
  write_to_bin(train_data_files, train_aspect_files, train_focus_files, train_label_files, os.path.join(finished_files_dir, "train.bin"), encode_type, True)
  # print('start writing dev file...')
#   write_to_bin(valid_data_files, valid_label_files, os.path.join(finished_files_dir, "dev.bin"), encode_type)
  # print('start writing test file...')
#   write_to_bin(test_data_files, test_label_files, os.path.join(finished_files_dir, "test.bin"), encode_type)

  # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
  chunk_all()
