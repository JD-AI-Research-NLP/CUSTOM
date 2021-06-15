root=$1
domain=$2
#catId=$3
catId=$3
folder=$4
#==================从KB中选取“家用电器”的所有sku的基本信息**JOIN**达人写作=========================
hive -e "select distinct 
a.item_sku_id,
a.sku_name,
a.brand_code,
a.brandname_en,
a.brandname_cn,
a.brandname_full,
a.item_first_cate_cd,
a.item_first_cate_name,
a.item_second_cate_cd,
a.item_second_cate_name,
a.item_third_cate_cd,
a.item_third_cate_name,
a.dt,
b.content_id,
b.content_pic,
b.label,
b.main_title,
b.description,
a.item_id
from 
(
select * from 
(
select
item_sku_id,
sku_name,
brand_code,
barndname_en as brandname_en,
barndname_cn as brandname_cn,
barndname_full as brandname_full,
item_first_cate_cd,
item_first_cate_name,
item_second_cate_cd,
item_second_cate_name,
item_third_cate_cd,
item_third_cate_name,
item_type,
dt,
ROW_NUMBER() OVER(PARTITION BY item_sku_id ORDER BY dt DESC) AS rowNum,
item_id
from gdm.gdm_m03_sold_item_sku_da
where dt=sysdate(-2) AND item_first_cate_cd=$catId AND sku_valid_flag=1) x
where rowNum=1) a
INNER JOIN 
(
select
content_id,
sku_uniq,
content_pic,
concat_ws(',',collect_set(label)) as label,
main_title,
description
from app.app_discovery_content_basic_info_da
lateral view explode(split(sku, \",\")) myTable as sku_uniq
where content_type=100 AND sub_position=8 AND NOT description like '\[%' and dt=sysdate(-2)
group by content_id,sku_uniq,content_pic,main_title,description
) b
ON a.item_sku_id=b.sku_uniq;" > $root/data/$domain'_'$folder/$domain".writing"

