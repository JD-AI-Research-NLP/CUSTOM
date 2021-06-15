ocr_file=$1
dir=$2
trainsku=$3
devsku=$4
testsku=$5
traintable=$6
devtable=$7
testtable=$8
echo $dir
echo $sku
echo $table
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
paste $trainsku $traintable > $dir/t
python3 comtable_2.py $dir/t $dir/out $dir/"train.table.com"
rm $dir/t
paste $devsku $devtable > $dir/t
python3 comtable_2.py $dir/t $dir/out $dir/"dev.table.com"
rm $dir/t
paste $testsku $testtable > $dir/t
python3 comtable_2.py $dir/t $dir/out $dir/"test.table.com"
rm $dir/t
rm $dir/out
