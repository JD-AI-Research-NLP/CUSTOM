root=$1/data
#"/export/homes/baojunwei/alpha-w-pipeline/data/"
domain=$2
#"chuju"
dataset=$3
#"test"

if [[ $dataset = 'train' ]];
then
 root1=$root/$domain
 inputSku=$dataset".sku"
else
 root1=$root/$domain"_"$dataset
 inputSku=$dataset".validSku"
fi

#root1=$root/$domain"_"$dataset
root7=$root/$domain
line_num=$(wc -l $root1/$inputSku)
let l=$line_num
thread_num=20
num=`expr $l / $thread_num`
echo $line_num
echo $num

split -l $num $root1/$inputSku $root1/skusplit
split -l $num $root1/$dataset".table" $root1/tablesplit
root2=$root/ckg/$domain/ckg1
root3=$root1/train.text
echo $root3

root4=$root/ckg/$domain/ckg1
root5=$root/$domain/train.title
for filesku in $root1/skusplit*
do
  #arr=$(echo $filesku|tr "/" "\n")
  OLD_IFS=$IFS
  IFS='/'
  arr=($filesku)
  IFS=$OLD_IFS
  
  filetable=$root1/"tablesplit"${arr[-1]:8}
  fileoutput=$root1/"outsplit"${arr[-1]:8}
  fileoutputtmp=$root1/"tmpoutsplit"${arr[-1]:8}
  ocr_file=$root1/$dataset".ocr"
  echo $ocr_file
  train_text=$root3
  nohup python prediction_step4_getOcrSP.py $filesku $filetable $fileoutput $fileoutputtmp $ocr_file $train_text $root4 $root5 $ocr_file"_"${arr[-1]:8} $root7/"train.table" &
  #nohup python prediction_step4_getOcrSP_2.py $filesku $filetable $fileoutput $fileoutputtmp $ocr_file $train_text $root4 $root5 $ocr_file  $root7/"train.table" &
done
wait
#rm $root1/tmp*
#rm $root1/skusplit*
#rm $root1/tablesplit*
cat $root1/outsplit* > $root1/$dataset".sp"
rm $root1/outsplit*
#rm $root1/$dataset'.ocr_'*
#rm $root1/outsplit*
#rm $root1/$dataset'.ocr_'*
for filesku in $root1/skusplit*
do
  #arr=$(echo $filesku|tr "/" "\n")
  OLD_IFS=$IFS
  IFS='/'
  arr=($filesku)
  IFS=$OLD_IFS

  filetable=$root1/"tablesplit"${arr[-1]:8}
  fileoutput=$root1/"outsplit"${arr[-1]:8}
  fileoutputtmp=$root1/"tmpoutsplit"${arr[-1]:8}
  ocr_file=$root1/$dataset".ocr"
  train_text=$root3
  nohup python prediction_step4_getOcrSP_2.py $filesku $filetable $fileoutput $fileoutputtmp $ocr_file $train_text $root4 $root5 $ocr_file"_"${arr[-1]:8} $root7/"train.table" &
  #python prediction_step4_getOcrSP_2.py $filesku $filetable $fileoutput $fileoutputtmp $ocr_file $train_text $root4 $root5 $ocr_file"_"${arr[-1]:8} $root7/"train.table"
done
wait
rm $root1/tmp*
rm $root1/skusplit*
rm $root1/tablesplit*
cat $root1/outsplit* > $root1/$dataset".sp_model_input"
rm $root1/outsplit*
rm $root1/$dataset'.ocr_'*
