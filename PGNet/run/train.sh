export CUDA_VISIBLE_DEVICES="1"
root="/home/liangjiahui8/notespace/Aspect_writing73"
domain="shouji"
dataset="train"
folder="rawData"
stop_step=50000
mkdir $root/data/$domain"_"$folder"_latest"
mv $root/data/cat_id $root/data/$domain"_"$folder"_latest"
mkdir $root/data/$domain"_"$folder
mkdir $root/data/$domain
dataset2=$domain"_dev"
if [ ! -d $root/data/$dataset2 ];then
mkdir $root/data/$dataset2
fi

#prediction_step6:================input2bin-writing==============================
#python3 train_step6_inputMakeBin_kb2writing_t2s.py $root $domain $dataset
#python3 prediction_step6_inputMakeBin_kb2writing_t2s.py $root $domain "dev" "table"
#exit

#prediction_step8:================writing generation==============================
sh train_step8_trainWritingGen.sh $root $domain $dataset $stop_step
exit


