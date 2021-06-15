export CUDA_VISIBLE_DEVICES="1"
root="/home/liangjiahui8/notespace/Aspect_writing"
# root="/export/scratch/wangyifan/kb_write2/test/shuma/shuma"
domain="shouji"
dataset="test"
beam_size=5
dataset2=$domain"_"$dataset
if [ ! -d $root/data/$dataset2 ];then
mkdir $root/data/$dataset2
fi
cp $root"/data/"$domain"/train.text" $root"/data/"$dataset2
maxSkuNum="200000"

folder="rawData_latest"
tableFile="table"
version="KB2.0"

eval_path=$root/log/$domain/kb2writing_t2s/eval/checkpoint
step=`head -n 1 $eval_path`
OLD_IFS="$IFS" ;IFS="-" ;arr=($step) ;IFS="$OLD_IFS"
tmp=${arr[-1]}; len=${#tmp}; modelStepsWriting=${tmp:0:$len-1}

#all_step1:================dump Data==============================
if [ ! -d $root/data/$domain"_"$folder ];then
mkdir $root/data/$domain"_"$folder
else
echo $root/data/$domain"_"$folder"文件夹已经存在"
fi

#prediction_step0:================get Sku==============================
#python3 prediction_step0_getSku.py $root $domain $dataset $maxSkuNum $version
#exit

#all_step1:================filter Data==============================
#python3 step1_getInfo_from_api.py $root $domain $dataset $folder
#exit

#prediction_step2:================get OCR==============================
#python3 prediction_step2_getOCR.py $root $domain $dataset $folder
#exit

#prediction_step3:================get validOCR==============================
#python3 prediction_step3_getValidOCR.py $root $domain $dataset
#exit

#prediction_step4:================get SP==============================
#sh prediction_step4_getOcrSP.sh $root $domain $dataset
#exit

#prediction_step5:================selectOcrSP==============================
#python3 prediction_step5_selectOcrSP.py $root $domain $dataset
#exit

#prediction_step5:================comtable==============================
#python3 prediction_step5_tableCompletion.py $root $domain $dataset
#exit

#prediction_step6:================input2bin-writing==============================
#python3 prediction_step6_inputMakeBin_kb2writing_t2s.py $root $domain $dataset $tableFile
#exit

#prediction_step8:================writing generation==============================
sh prediction_step8_writingGen.sh $root $domain $dataset $modelStepsWriting $beam_size
exit

#prediction_step9:================title generation==============================
python3 prediction_step9_titleGen.py $root $domain $folder $dataset 
exit

#prediction_step10:================merge_data==============================
python3 merge.py $root $domain 20 $dataset
exit

#prediction_step10:================merge_data==============================
python3 filterByCat3.py $root $domain 20 $dataset
#exit

#prediction_step9:================upload==============================
#python3 upload.py $root $domain $dataset
#exit

#prediction_step10:================highlight==============================
python3 highlight.py $root $domain $dataset
#exit
