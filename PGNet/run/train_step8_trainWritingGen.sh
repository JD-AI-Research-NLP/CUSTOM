MyHome=$1 #/export/homes/baojunwei/alpha-w-pipeline
domain=$2
dataset=$3 #train
stop_step=$4

# touch $MyHome/data/trigram/$domain/source.dict
touch $MyHome/data/trigram/$domain/train_dic

#==========t2s-GridBS===========
python3 $MyHome/src_t2s_GridBS/run_summarization.py --sp_word_mask=$MyHome/data/trigram/$domain/source.dict.copy --source_dict=$MyHome/data/trigram/$domain/source.dict --train_text_path=$MyHome/data/trigram/$domain/train_dic --arg1=0.3 --arg2=0.7 --trigram_v2_path=$MyHome/data/trigram/$domain/trigram --mode=train --data_path=$MyHome/data/$domain"_train_writing_bin"/chunked/train_* --eval_data_path=$MyHome/data/$domain"_dev_writing_bin"/chunked/* --vocab_aspect_path=$MyHome/data/$domain"_train_writing_bin"/bin/vocab_aspect --vocab_path=$MyHome/data/$domain"_train_writing_bin"/bin/vocab --ckg_path=$MyHome/data/ckg/$domain/ckg0/* --blackWord_path=$MyHome/data/blackList.txt --constraint_path=$MyHome/data/$domain/train.fakeConstraint --eval_constraint_path=$MyHome/data/$domain"_dev"/dev.constraint-5sellingPoints-ocr-largeIdf --ocr_path=$MyHome/data/$domain/train.fakeValidOcr --eval_ocr_path=$MyHome/data/$domain"_dev"/dev.validOcr  --log_root=$MyHome/log/$domain --exp_name=kb2writing_t2s --stopWords_path=$MyHome/data/stopword.hit --dataset=$dataset --trigram_path=$MyHome/data/trigram/$domain/triletter --steps_halve=3000 --steps_kill=10 --unigram_path=$MyHome/data/biLM1.pkl --bigram_path=$MyHome/data/biLM2.pkl --conflic_path=$MyHome/data/conflicValues
#pid1=$!
#sleep 100s

#dataset="dev"
#python3 $MyHome/src_t2s_GridBS/run_summarization_eval.py --sp_word_mask=$MyHome/data/trigram/$domain/source.dict.copy --stop_step=$stop_step --eval_text=$MyHome/data/$domain/dev.text --pid=$pid1 --source_dict=$MyHome/data/trigram/$domain/source.dict --train_text_path=$MyHome/data/trigram/$domain/train_dic --arg1=0.3 --arg2=0.7 --mode=decode --trigram_v2_path=$MyHome/data/trigram/$domain/trigram --trigram_path=$MyHome/data/trigram/$domain/triletter --data_path=$MyHome/data/$domain'_'$dataset'_writing_bin'/chunked/test_* --vocab_path=$MyHome/data/$domain'_train_writing_bin'/bin/vocab --ckg_path=$MyHome/data/ckg/$domain/ckg0/* --blackWord_path=$MyHome/data/blackList.txt --constraint_path=$MyHome/data/$domain'_'$dataset/$dataset'.constraint-5sellingPoints-ocr-largeIdf' --log_root=$MyHome/log/$domain/kb2writing_t2s --exp_name=$exp --single_pass=1 --force_decoding=True  --stopWords_path=$MyHome/data/stopword.hit --decode_id_path=$MyHome/data/$domain'_'$dataset/$dataset'.validSku.flag' --funcWords_file_path=$MyHome/data/funcWord/$domain/* --ocr_path=$MyHome/data/$domain'_'$dataset/$dataset'.validOcr' --dataset=$dataset
