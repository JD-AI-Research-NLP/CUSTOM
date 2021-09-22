MyHome=$1 
domain=$2 
dataset=$3
modelStepsWriting=$4
beam_size=$5

rm -r $MyHome/log/$domain/'kb2writing_t2s/decode_'$dataset'_200maxenc_'$beam_size'beam_20mindec_70maxdec_force_decoding_ckg_gridBS_5sellPoint_ocr_funcWords_noRedundentUnigram_upgradeConsistent_ckpt-'$modelStepsWriting'0.3_0.7'

python3 $MyHome/src_t2s_GridBS/run_summarization.py \
--train_text_path=$MyHome/data/trigram/$domain/train_dic \
--arg1=0.3 \
--arg2=0.7 \
--mode=decode \
--data_path=$MyHome/data/$domain'_'$dataset'_writing_bin'/chunked/test_* \
--vocab_path=$MyHome/data/$domain'_train_writing_bin'/bin/vocab \
--vocab_aspect_path=$MyHome/data/$domain'_train_writing_bin'/bin/vocab_aspect \
--log_root=$MyHome/log/$domain/kb2writing_t2s \
--single_pass=1 \
--force_decoding=True  \
--dataset=$dataset \
--beam_size=$beam_size \
wait
cp $MyHome/log/$domain/'kb2writing_t2s/decode_'$dataset'_200maxenc_'$beam_size'beam_20mindec_70maxdec_force_decoding_ckg_gridBS_5sellPoint_ocr_funcWords_noRedundentUnigram_upgradeConsistent_ckpt-'$modelStepsWriting'0.3_0.7/'decoded/decoded.txt $MyHome/data/$domain'_'$dataset/$dataset'.prediction.writing'

