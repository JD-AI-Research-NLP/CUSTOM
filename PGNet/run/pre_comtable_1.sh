ocr_file=$1
dir=$2
sku=$3
table=$4
split -l 100000 $ocr_file $dir/split
for file in $dir/split*
do
  nohup python3 comtable_1.py $file $dir &
done
wait
cat $dir/*_out > $dir/final
rm $dir/*_out
rm $dir/split* 
sort -t $'\t' -k 1nr -k 4nr -k 2nr -k 3nr $dir/final > $dir/out
rm $dir/final
paste $sku $table > $dir/t
python3 comtable_2.py $dir/t $dir/out $dir/out1
rm $dir/t
rm $dir/out
