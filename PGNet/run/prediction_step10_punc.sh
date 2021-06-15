#root='/export/homes/baojunwei/alpha-w-pipeline'
root=$1
#domain='chuju'
domain=$2
#dataset='test'
dataset=$3
input=$root/data/$domain"_"$dataset/$dataset".prediction.writing.ori"
cd $root/run/punc_model_1
python3 punc_step1.py $input
nohup sh run.sh writing_to_writing &
sh run_test.sh writing_to_writing > done
wait
#python3 punc_step2.py
python3 punc_step3.py
mv final $root/data/$domain'_'$dataset/$dataset'.prediction.writing'
cd ../

