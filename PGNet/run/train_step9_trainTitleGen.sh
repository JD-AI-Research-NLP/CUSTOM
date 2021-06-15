MyHome=$1 #/export/homes/baojunwei/alpha-w-pipeline
domain=$2 #'chuju'
dataset=$3 #'train'
#==========s2s===========
#python3 ./src_s2s/run_summarization.py --mode=train --data_path=./data/kb2writing_seq2seq/chunked/train_* --vocab_path=./data/kb2writing_seq2seq/bin/vocab --log_root=./log/jiayongdianqi --exp_name=kb2writing_s2s

#==========s2s-title===========
CUDA_VISIBLE_DEVICES=2 python3 $MyHome/src_s2s/run_summarization.py --mode=train --data_path=$MyHome/data/$domain'_train_title_bin'/chunked/train_* --vocab_path=$MyHome/data/$domain'_train_title_bin'/bin/vocab --log_root=$MyHome/log/$domain --exp_name=kb2title_s2s --dataset=$dataset

#==========s2s-coverage===========
#python3 ./src_s2s/run_summarization.py --mode=train --data_path=./data/kb2writing_seq2seq/chunked/train_* --vocab_path=./data/kb2writing_seq2seq/bin/vocab --log_root=./log/jiayongdianqi --exp_name=kb2writing_s2s_cov --coverage=True --convert_to_coverage_model=True
 
#==========t2s-coverage===========
#python3 ./src_t2s/run_summarization.py --mode=train --data_path=./data/kb2writing_table2seq_bigVocab/chunked/train_* --vocab_path=./data/kb2writing_table2seq_bigVocab/bin/vocab --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_cov --coverage=True --convert_to_coverage_model=True

#==========t2s-v1===========
#python3 ./src_t2s_v1/run_summarization.py --mode=train --data_path=./data/kb2writing_table2seq_bigVocab/chunked/train_* --vocab_path=./data/kb2writing_table2seq_bigVocab/bin/vocab --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab
#python3 ./src_t2s_v1/run_summarization.py --mode=train --data_path=./data/kb2writing_table2seq_controlVocab/chunked/train_* --vocab_path=./data/kb2writing_table2seq_controlVocab/bin/vocab --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_controlVocab

#==========t2s===========
#python3 ./src_t2s/run_summarization.py --mode=train --data_path=./data/kb2writing_table2seq_bigVocab/chunked/train_* --vocab_path=./data/kb2writing_table2seq_bigVocab/bin/vocab --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s
#python3 ./src_t2s/run_summarization.py --mode=train --data_path=./data/kb2writing_table2seq_controlVocab/chunked/train_* --vocab_path=./data/kb2writing_table2seq_controlVocab/bin/vocab --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_controlVocab

#==========t2s-GridBS===========
#python3 ./src_t2s_GridBS/run_summarization.py --mode=train --data_path=./data/kb2writing_table2seq_filtered_bigVocab/chunked/train_* --vocab_path=./data/kb2writing_table2seq_filtered_bigVocab/bin/vocab --ckg_path=./data/ckg/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/jiayongdianqi_filtered/train.fakeConstraint --log_root=./log/jiayongdianqi_filtered --exp_name=kb2writing_t2s_filtered_bigVocab --stopWords_path=./data/black_list/stopword.hit

#python3 $MyHome/src_t2s_GridBS/run_summarization.py --mode=train --data_path=$MyHome/data/kb2writing_t2s/chunked/train_* --vocab_path=$MyHome/data/kb2writing_t2s/bin/vocab --ckg_path=$MyHome/data/ckg/chuju/* --blackWord_path=$MyHome/data/blackList.txt --constraint_path=$MyHome/data/chuju/train.fakeConstraint --ocr_path=$MyHome/data/chuju/train.fakeValidOcr  --log_root=$MyHome/log/chuju --exp_name=kb2writing_t2s --stopWords_path=$MyHome/data/stopword.hit
