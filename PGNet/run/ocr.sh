root=$1
input=$2
output=$3
#rm $root/sku_itemid_ocr
line_num=$(wc -l $input)
let l=$line_num
thread_num=20
num=`expr $l / $thread_num`
echo $num
split -l $num $input $root/split
ls -ltr
for file1 in $root/split*
do
  echo $file1
  nohup python3 img_ocr.py $file1 &
done
wait
cat $root/split*_out | sort > $output
rm $root/split*

