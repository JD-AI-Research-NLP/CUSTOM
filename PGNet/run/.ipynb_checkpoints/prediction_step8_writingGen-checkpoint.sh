#===============t2s-forceD========================
#python3 ./src_t2s_v1/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin/chunked/test_* --vocab_path=./data/kb2writing_table2seq_xuanpin/bin/vocab --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_controlVocab --single_pass=1 --force_decoding=True

#python3 ./src_t2s_v1/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin/chunked/test_* --vocab_path=./data/kb2writing_table2seq_xuanpin/bin/vocab --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True

#===============t2s-forceD-CKG========================
#python3 ./src_t2s/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin/chunked/test_* --vocab_path=./data/kb2writing_table2seq_xuanpin/bin/vocab --ckg_path=./analysis/groupByCat3_withValue/* --blackWord_path=./data/black_list/blackList.txt --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True

#python3 ./src_t2s/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin/chunked/test_* --vocab_path=./data/kb2writing_table2seq_xuanpin/bin/vocab --ckg_path=./data/ckg/* --blackWord_path=./data/black_list/blackList.txt --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True


#===============t2s-forceD-CKG-GridBS========================
#xuanpin---------------------------------------------
#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin/chunked/test_* --vocab_path=./data/kb2writing_table2seq_xuanpin/bin/vocab --ckg_path=./data/ckg/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/xuanpin/xuanpin.merge.sku.constraint.new-2sellingPoints --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True

#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin/chunked/test_* --vocab_path=./data/kb2writing_table2seq_xuanpin/bin/vocab --ckg_path=./data/ckg/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/xuanpin/xuanpin.merge.sku.constraint.new-3sellingPoints --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True

#xuanpin2---------------------------------------------
#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin2/chunked/test_* --vocab_path=./data/kb2writing_table2seq_xuanpin2/bin/vocab --ckg_path=./data/ckg/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/xuanpin2/xuanpin2.constraint-2sellingPoints --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True

#xuanpin3---------------------------------------------
#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin3/chunked/test_* --vocab_path=./data/kb2writing_table2seq_xuanpin3/bin/vocab --ckg_path=./data/ckg/* --blackWord_path=./data/black_list/blackList.forGeneration.txt --constraint_path=./data/xuanpin3/xuanpin3.constraint-2sellingPoints --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True

#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin3/chunked/test_* --vocab_path=./data/kb2writing_table2seq_xuanpin3/bin/vocab --ckg_path=./data/ckg/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/xuanpin3/xuanpin3.constraint-2sellingPoints --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True

#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin3/chunked/test_* --vocab_path=./data/kb2writing_table2seq_xuanpin3/bin/vocab --ckg_path=./data/ckg/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/xuanpin3/xuanpin3.constraint-1sellingPoints-largeIdf --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True

#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin3/chunked/test_* --vocab_path=./data/kb2writing_table2seq_xuanpin3/bin/vocab --ckg_path=./data/ckg/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/xuanpin3/xuanpin3.constraint-1sellingPoints-smallIdf --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True

#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin3/chunked/test_* --vocab_path=./data/kb2writing_table2seq_xuanpin3/bin/vocab --ckg_path=./data/ckg/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/xuanpin3/xuanpin3.constraint-0sellingPoints --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True

#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin3/chunked/test_* --vocab_path=./data/kb2writing_table2seq_xuanpin3/bin/vocab --ckg_path=./data/ckg/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/xuanpin3/xuanpin3.constraint-2sellingPoints --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True --decode_id_path=./data/xuanpin3/xuanpin3.skus.300.flag

#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin3/chunked/test_* --vocab_path=./data/kb2writing_table2seq_xuanpin3/bin/vocab --ckg_path=./data/ckg/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/xuanpin3/xuanpin3.constraint-2sellingPoints --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True --decode_id_path=./data/xuanpin3/xuanpin3.skus.300.flag --stopWords_path=./data/black_list/stopword.hit

#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin3/chunked/test_* --vocab_path=./data/kb2writing_table2seq_xuanpin3/bin/vocab --ckg_path=./data/ckg/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/xuanpin3/xuanpin3.constraint-1sellingPoints-ocr-largeIdf --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True --decode_id_path=./data/xuanpin3/xuanpin3.skus.300.flag --stopWords_path=./data/black_list/stopword.hit

#xuanpin4---------------------------------------------
#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin4/chunked/test_* --vocab_path=./data/kb2writing_table2seq_xuanpin4/bin/vocab --ckg_path=./data/ckg/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/xuanpin4/xuanpin4.constraint-3sellingPoints-ocr-largeIdf --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True --decode_id_path=./data/xuanpin4/xuanpin4.validSku.flag --stopWords_path=./data/black_list/stopword.hit

#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin4/chunked/test_* --vocab_path=./data/kb2writing_table2seq_bigVocab/bin/vocab --ckg_path=./analysis/groupByCat3_withValue.5Cat/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/xuanpin4/xuanpin4.constraint-3sellingPoints-ocr-largeIdf --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True --decode_id_path=./data/xuanpin4/xuanpin4.validSku.flag --stopWords_path=./data/black_list/stopword.hit

#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin4/chunked/test_* --vocab_path=./data/kb2writing_table2seq_bigVocab/bin/vocab --ckg_path=./analysis/groupByCat3_withValue.5Cat/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/xuanpin4/xuanpin4.constraint-3sellingPoints-ocr-largeIdf --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True --decode_id_path=./data/xuanpin4/xuanpin4.validSku.flag --stopWords_path=./data/black_list/stopword.hit --funcWords_file_path=./data/funcWords/* --ocr_path=./data/xuanpin4/xuanpin4.validOcr

#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin4/chunked/test_* --vocab_path=./data/kb2writing_table2seq_bigVocab_v2/bin/vocab --ckg_path=./analysis/groupByCat3_withValue.5Cat/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/xuanpin4/xuanpin4.constraint-3sellingPoints-ocr-largeIdf --log_root=./log/jiayongdianqi_v2 --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True --decode_id_path=./data/xuanpin4/xuanpin4.validSku.flag --stopWords_path=./data/black_list/stopword.hit --funcWords_file_path=./data/funcWords/* --ocr_path=./data/xuanpin4/xuanpin4.validOcr


#xuanpin6---------------------------------------------
#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin6/chunked/test_* --vocab_path=./data/kb2writing_table2seq_bigVocab/bin/vocab --ckg_path=./analysis/groupByCat3_withValue.5Cat/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/xuanpin6/xuanpin6.constraint-3sellingPoints-ocr-largeIdf --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True --decode_id_path=./data/xuanpin6/xuanpin6.validSku.flag --stopWords_path=./data/black_list/stopword.hit

#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin6/chunked/test_* --vocab_path=./data/kb2writing_table2seq_bigVocab/bin/vocab --ckg_path=./analysis/groupByCat3_withValue.5Cat/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/xuanpin6/xuanpin6.constraint-3sellingPoints-ocr-largeIdf --log_root=./log/jiayongdianqi --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True --decode_id_path=./data/xuanpin6/xuanpin6.validSku.flag --stopWords_path=./data/black_list/stopword.hit --funcWords_file_path=./data/funcWords/* --ocr_path=./data/xuanpin6/xuanpin6.validOcr

#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_xuanpin6/chunked/test_* --vocab_path=./data/kb2writing_table2seq_bigVocab_v2/bin/vocab --ckg_path=./analysis/groupByCat3_withValue.5Cat/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/xuanpin6/xuanpin6.constraint-3sellingPoints-ocr-largeIdf --log_root=./log/jiayongdianqi_v2 --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True --decode_id_path=./data/xuanpin6/xuanpin6.validSku.flag --stopWords_path=./data/black_list/stopword.hit --funcWords_file_path=./data/funcWords/* --ocr_path=./data/xuanpin6/xuanpin6.validOcr


#alphaSales---------------------------------------------
#python3 ./src_t2s_GridBS/run_summarization.py --mode=decode --data_path=./data/kb2writing_table2seq_alphaSales/chunked/test_* --vocab_path=./data/kb2writing_table2seq_bigVocab_v2/bin/vocab --ckg_path=./analysis/groupByCat3_withValue/* --blackWord_path=./data/black_list/blackList.txt --constraint_path=./data/alphaSales/alphaSales.constraint-3sellingPoints-ocr-largeIdf --log_root=./log/jiayongdianqi_v2 --exp_name=kb2writing_t2s_bigVocab --single_pass=1 --force_decoding=True --decode_id_path=./data/alphaSales/alphaSales.validSku.flag --stopWords_path=./data/black_list/stopword.hit --funcWords_file_path=./data/funcWords/* --ocr_path=./data/alphaSales/alphaSales.validOcr

MyHome=$1 #/export/homes/baojunwei/alpha-w-pipeline
domain=$2 #'chuju'
dataset=$3 #'test'
modelStepsWriting=$4
beam_size=$5
rm -r $MyHome/log/$domain/'kb2writing_t2s/decode_'$dataset'_200maxenc_'$beam_size'beam_20mindec_70maxdec_force_decoding_ckg_gridBS_5sellPoint_ocr_funcWords_noRedundentUnigram_upgradeConsistent_ckpt-'$modelStepsWriting'0.3_0.7'

# python3 $MyHome/src_t2s_GridBS/run_summarization.py --sp_word_mask=$MyHome/data/trigram/$domain/source.dict.copy --source_dict=$MyHome/data/trigram/$domain/source.dict --train_text_path=$MyHome/data/trigram/$domain/train_dic --arg1=0.3 --arg2=0.7 --mode=decode --trigram_v2_path=$MyHome/data/trigram/$domain/trigram --trigram_path=$MyHome/data/trigram/$domain/triletter --data_path=$MyHome/data/$domain'_'$dataset'_writing_bin'/chunked/test_* --vocab_path=$MyHome/data/$domain'_train_writing_bin'/bin/vocab --ckg_path=$MyHome/data/ckg/$domain/ckg0/* --blackWord_path=$MyHome/data/blackList.txt --constraint_path=$MyHome/data/$domain'_'$dataset/$dataset'.constraint-5sellingPoints-ocr-largeIdf' --log_root=$MyHome/log/$domain/kb2writing_t2s  --single_pass=1 --force_decoding=True  --stopWords_path=$MyHome/data/stopword.hit --decode_id_path=$MyHome/data/$domain'_'$dataset/$dataset'.validSku.flag' --funcWords_file_path=$MyHome/data/funcWord/$domain/* --ocr_path=$MyHome/data/$domain'_'$dataset/$dataset'.validOcr' --dataset=$dataset  --beam_size=$beam_size
python3 $MyHome/src_t2s_GridBS/run_summarization.py \
--sp_word_mask=$MyHome/data/trigram/$domain/source.dict.copy \
--source_dict=$MyHome/data/trigram/$domain/source.dict \
--train_text_path=$MyHome/data/trigram/$domain/train_dic \
--arg1=0.3 \
--arg2=0.7 \
--mode=decode \
--trigram_v2_path=$MyHome/data/trigram/$domain/trigram \
--trigram_path=$MyHome/data/trigram/$domain/triletter \
--data_path=$MyHome/data/$domain'_'$dataset'_writing_bin'/chunked/test_* \
--vocab_path=$MyHome/data/$domain'_train_writing_bin'/bin/vocab \
--vocab_aspect_path=$MyHome/data/$domain'_train_writing_bin'/bin/vocab_aspect \
--ckg_path=$MyHome/data/ckg/$domain/ckg0/* \
--blackWord_path=$MyHome/data/blackList.txt \
--constraint_path=$MyHome/data/$domain'_'$dataset/$dataset'.constraint-5sellingPoints-ocr-largeIdf' \
--log_root=$MyHome/log/$domain/kb2writing_t2s \
--single_pass=1 \
--force_decoding=True  \
--stopWords_path=$MyHome/data/stopword.hit \
--decode_id_path=$MyHome/data/$domain'_'$dataset/$dataset'.validSku.flag' \
--funcWords_file_path=$MyHome/data/funcWord/$domain/* \
--ocr_path=$MyHome/data/$domain'_'$dataset/$dataset'.validOcr' \
--dataset=$dataset \
--beam_size=$beam_size \
--unigram_path=$MyHome/data/biLM1.pkl \
--bigram_path=$MyHome/data/biLM2.pkl \
--conflic_path=$MyHome/data/conflicValues

wait

cp $MyHome/log/$domain/'kb2writing_t2s/decode_'$dataset'_200maxenc_'$beam_size'beam_20mindec_70maxdec_force_decoding_ckg_gridBS_5sellPoint_ocr_funcWords_noRedundentUnigram_upgradeConsistent_ckpt-'$modelStepsWriting'0.3_0.7/'decoded/decoded.txt $MyHome/data/$domain'_'$dataset/$dataset'.prediction.writing'

