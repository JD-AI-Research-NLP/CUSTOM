export CUDA_VISIBLE_DEVICES="3"
root="" # full root dir of project
domain="shouji"
dataset="test"
beam_size=5
tokenizer="pretrained_model" # pretrained bert-base chinese version dir
dataset2=$domain"_"$dataset
if [ ! -d $root/data/$dataset2 ];then
mkdir $root/data/$dataset2
fi
eval_path=$root/log/$domain/kb2writing_t2s/train/checkpoint
step=`head -n 1 $eval_path`
arr=($step)
tmp=${arr[-1]}; len=${#tmp}; modelStepsWriting=${tmp:0:$len-1}

#prediction_step1:================input2bin-writing==============================
python3 prediction_inputMakeBin_kb2writing_t2s.py $root $domain $dataset $tokenizer
#exit

#prediction_step2:================writing generation==============================
sh prediction_writingGen.sh $root $domain $dataset $modelStepsWriting $beam_size
exit