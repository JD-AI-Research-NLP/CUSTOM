export CUDA_VISIBLE_DEVICES="1"
root="./PGNet" # full root dir of project
domain="shouji" # shouji or diannao
dataset="train"
folder="rawData"
tokenizer="pretrained_model" # pretrained bert-base chinese version dir
if [ ! -d $root/data/$domain ];then
    mkdir $root/data/$domain
fi
dataset2=$domain"_dev"
if [ ! -d $root/data/$dataset2 ];then
mkdir $root/data/$dataset2
fi
python3 train_inputMakeBin_kb2writing_t2s.py $root $domain $dataset $tokenizer
python3 prediction_inputMakeBin_kb2writing_t2s.py $root $domain "dev" $tokenizer
sh train_trainWritingGen.sh $root $domain $dataset
exit


