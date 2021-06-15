root="/export/homes/baojunwei/alpha-w-pipeline"
#root="/data0/baojunwei/alpha-w-pipeline"
#domain="chuju"
domain="jiayongdianqi"
dataset="xuanpin6bad"
#dataset="xuanpin7"
maxSkuNum="200"
catId="737"
#catId="6196"
folder="rawData_latest"
writingTableFile="table.comp"
titleTableFile="table"
version="KB1.0"
#cat3list=("茶壶" "保温杯" "玻璃杯" "茶杯")
#cat3list=("冰箱" "洗衣机" "净水器" "油烟机" "空调" "平板电视" "燃气热水器" "扫地机器人" "口腔护理" "电热水器" "空气净化器" "吸尘器" "美容器" "按摩器" "电饭煲" "电水壶/热水瓶" "破壁机" "按摩椅")
catIDs=("870" "1300" "878" "880" "898")
#modelStepsWriting="678750" #chuju
modelStepsWriting="1135857" #jiayongdianqi
#modelStepsTitle="279580" #chuju
modelStepsTitle="244728" #jiayongdianqi

writingExp="kb2writing_t2s_compTable"
titleExp="kb2title_s2s"
echo $root
echo $domain
echo $dataset
echo $catId
echo $folder
#echo ${cat3list[0]}
echo $modelStepsWriting
echo $modelStepsTitle

#all_step1:================dump Data==============================
if [ ! -d $root/data/$domain"_"$folder ];then
mkdir $root/data/$domain"_"$folder
else
echo $root/data/$domain"_"$folder"文件夹已经存在"
fi

#nohup sh $root/run/all_step0_dumpData_3.sh $root $domain $catId $folder &
#for script in $root/run/all_step0_dumpData_*.sh
#do
#  nohup sh $script $root $domain $catId $folder 1>$script'.log' 2>&1 &
#done
#wait

#all_step2:================filter Data==============================
#python3 all_step2_filterData.py $root $domain $folder

#prediction_step0:================get Sku==============================
#python3 prediction_step0_getSku.py $root $domain $dataset $maxSkuNum $version ${catIDs[*]}

#prediction_step1:================get KB==============================
#python3 prediction_step1_getKB.py $root $domain $dataset $folder #${cat3list[*]}

#prediction_step2:================get OCR==============================
#python3 prediction_step2_getOCR.py $root $domain $dataset $folder

#prediction_step3:================get validOCR==============================
#python3 prediction_step3_getValidOCR.py $root $domain $dataset

#prediction_step4:================get SP==============================
#sh prediction_step4_getOcrSP.sh $root $domain $dataset

#prediction_step5:================get completed table==============================
#python3 prediction_step5_tableCompletion.py $root $domain $dataset

#prediction_step5:================selectOcrSP==============================
#python3 prediction_step5_selectOcrSP.py $root $domain $dataset

#prediction_step6:================input2bin-writing==============================
python3 prediction_step6_inputMakeBin_kb2writing_t2s.py $root $domain $dataset $writingTableFile

#prediction_step7:================input2bin-title==============================
python3 prediction_step7_inputMakeBin_kb2title_s2s.py $root $domain $dataset $titleTableFile

#prediction_step8:================writing generation==============================
sh prediction_step8_writingGen.sh $root $domain $dataset $modelStepsWriting $writingExp

#prediction_step9:================title generation==============================
sh prediction_step9_titleGen.sh $root $domain $dataset $modelStepsTitle $titleExp

#prediction_step10:================rewriting==============================
sh prediction_step10_punc.sh $root $domain $dataset

#prediction_step11:================discriminator==============================
python3 prediction_step11_discriminate.py $root $domain $dataset

#prediction_step12:================post processing==============================
python3 prediction_step12_postPrcessing.py $root $domain $dataset $version

#prediction_step13:================join pic==============================
python3 prediction_step13_joinPic.py $root $domain $dataset $version


