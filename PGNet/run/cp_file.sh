root=$1
domain=$2
dataset2=$3
cp $root"/data/"$domain"/train.text" $root"/data/"$dataset2
cp $root"/data/"$domain"/dev.sku" $root"/data/"$dataset2"/dev"
cp $root"/data/"$domain"/dev.sku" $root"/data/"$dataset2"/dev.validSku"
cp $root"/data/"$domain"/dev.fakewriting" $root"/data/"$dataset2"/dev.validOcr"
cp $root"/data/"$domain"/dev.text" $root"/data/"$dataset2"/dev.fakeWriting"
cp $root"/data/"$domain"/dev.validSku.flag" $root"/data/"$dataset2"/dev.validSku.flag"
cp $root"/data/"$domain"/dev.fakewriting" $root"/data/"$dataset2"/dev.constraint-5sellingPoints-ocr-largeIdf"
cp $root"/data/"$domain"/dev.table.com" $root"/data/"$dataset2"/out1"
cp $root"/data/"$domain"/dev.aspect" $root"/data/"$dataset2"/dev.aspect"
cp $root"/data/"$domain"/dev.table" $root"/data/"$dataset2"/dev.table"
