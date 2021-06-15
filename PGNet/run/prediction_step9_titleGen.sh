#python3 ./src_s2s/run_summarization.py --mode=decode --data_path=./data/kb2title_seq2seq_xuanpin/chunked/test_* --vocab_path=./data/kb2title_seq2seq_xuanpin/bin/vocab --log_root=./log/jiayongdianqi --exp_name=kb2title_s2s --single_pass=1 --force_decoding=True

#python3 ./src_s2s/run_summarization.py --mode=decode --data_path=./data/kb2title_seq2seq_xuanpin2/chunked/test_* --vocab_path=./data/kb2title_seq2seq_xuanpin2/bin/vocab --log_root=./log/jiayongdianqi --exp_name=kb2title_s2s --single_pass=1 --force_decoding=True

#python3 ./src_s2s/run_summarization.py --mode=decode --data_path=./data/kb2title_seq2seq_xuanpin3/chunked/test_* --vocab_path=./data/kb2title_seq2seq_xuanpin3/bin/vocab --log_root=./log/jiayongdianqi --exp_name=kb2title_s2s --single_pass=1 --force_decoding=True

#python3 ./src_s2s/run_summarization.py --mode=decode --data_path=./data/kb2title_seq2seq_xuanpin4/chunked/test_* --vocab_path=./data/kb2title_seq2seq_xuanpin4/bin/vocab --log_root=./log/jiayongdianqi --exp_name=kb2title_s2s --single_pass=1 --force_decoding=True

#python3 ./src_s2s/run_summarization.py --mode=decode --data_path=./data/kb2title_seq2seq_xuanpin6/chunked/test_* --vocab_path=./data/kb2title_seq2seq/bin/vocab --log_root=./log/jiayongdianqi --exp_name=kb2title_s2s --single_pass=1 --force_decoding=True

MyHome=$1 #/export/homes/baojunwei/alpha-w-pipeline
domain=$2 #'chuju'
dataset=$3 #'test'
modelStepsTitle=$4
exp=$5
python3 $MyHome/src_s2s/run_summarization.py --mode=decode --data_path=$MyHome/data/$domain'_'$dataset'_title_bin'/chunked/test_* --vocab_path=$MyHome/data/$domain'_train_title_bin'/bin/vocab --log_root=$MyHome/log/$domain --exp_name=$exp --single_pass=1 --force_decoding=True --dataset=$dataset
cp $MyHome/log/$domain/$exp/'decode_'$dataset'_400maxenc_4beam_3mindec_50maxdec_force_decoding_ckpt-'$modelStepsTitle/decoded/decoded.txt $MyHome/data/$domain'_'$dataset/$dataset'.prediction.title'

