export CUDA_VISIBLE_DEVICES="1"
root="/home/liangjiahui8/notespace/Aspect_writing73"
# root="/export/scratch/wangyifan/kb_write2/test/shuma/shuma"
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

#all_step0:======================Get Info Data=============================
#python3 train_step0_getInfo_from_hiveTable.py $root $domain $folder
#exit

#prediction_step0:================filter Data==============================
#python3 all_step2_filterData.py $root $domain $folder
#exit

#prediction_step1:================dump writing data==============================
#python3 train_step1_dump_writing_data.py $root $domain $folder
#exit

#prediction_step2:================split train/dev/test==============================
#python3 train_step2_split_train_dev_test.py $root $domain $folder
#exit

#prediction_step2:================split train/dev/test==============================
#python3 train_step2_get_wordScore_from_text.py $root $domain $dataset $folder

#prediction_step3:================get OCR==============================
#python3 prediction_step2_getOCR.py $root $domain $dataset $folder

#prediction_step3:================dump ckg0==============================
#python3 train_step3_ckg0.py $root $domain $folder
#exit

##prediction_step4:================dump ckg1==============================
#python3 train_step4_ckg1.py $root $domain
#exit

#prediction_step5:================dump funcWord==============================
#python3 train_step5_funcW.py $root $domain $dataset $folder ${cat3list[*]}
#exit

#prediction_step6:================dump triGram==============================
#python3 train_step6_triletter.py $root $domain
#exit

#prediction_step4:================table comp==============================
#python3 train_step6_tableCompletion.py $root $domain $dataset
#python3 train_step6_refine_table.py $root $domain $dataset
#sh cp_file.sh $root $domain $dataset2
#exit

#prediction_step6:================input2bin-writing==============================
#python3 train_step6_inputMakeBin_kb2writing_t2s.py $root $domain $dataset
#python3 prediction_step6_inputMakeBin_kb2writing_t2s.py $root $domain "dev" "table"
#exit
#prediction_step7:================process-title==============================
#python3 train_step7_title_process.py $root $domain $folder 
#python3 train_step7_sourceDict.py $root $domain $folder
#exit #标注
#python3 train_step7_sourceDictCopyLabel.py $root $domain $folder
#exit #标注
#python3 train_step7_sourceDictCopy.py $root $domain $folder
#exit #标注
#python3 train_step7_buildBiGramLM.py $root $domain $folder
#python3 train_step7_getSupportedCats.py $root $domain $folder 
#exit #标注 check

#prediction_step8:================writing generation==============================
sh train_step8_trainWritingGen.sh $root $domain $dataset $stop_step
exit


