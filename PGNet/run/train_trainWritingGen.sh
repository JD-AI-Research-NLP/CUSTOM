MyHome=$1 #/export/homes/baojunwei/alpha-w-pipeline
domain=$2
dataset=$3 #train

# touch $MyHome/data/trigram/$domain/source.dict
touch $MyHome/data/trigram/$domain/train_dic

#==========t2s-GridBS===========
python3 $MyHome/src_t2s_GridBS/run_summarization.py --train_text_path=$MyHome/data/trigram/$domain/train_dic --arg1=0.3 --arg2=0.7 --mode=train --data_path=$MyHome/data/$domain"_train_writing_bin"/chunked/train_* --eval_data_path=$MyHome/data/$domain"_dev_writing_bin"/chunked/* --vocab_aspect_path=$MyHome/data/$domain"_train_writing_bin"/bin/vocab_aspect --vocab_path=$MyHome/data/$domain"_train_writing_bin"/bin/vocab  --log_root=$MyHome/log/$domain --exp_name=kb2writing_t2s --dataset=$dataset --steps_halve=3000 --steps_kill=10 
#pid1=$!
#sleep 100s