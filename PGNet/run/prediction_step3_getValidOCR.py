import sys
import os

def filter_kb_spec(input_file):
  with open(input_file, 'r', encoding='utf-8') as f:
    with open(input_file+'.filtered.concat', 'w', encoding='utf-8') as w:
      errCnt = 0
      while(True):
        line = f.readline()
        if not line:
          break
        line = line.strip('\n')
        parts = line.split('\t')
        sku = parts[0]
        r1 = parts[2]
        r2 = parts[4]
        v = parts[6]
        v_rem = parts[7]
        v_unit = parts[8]
        v_alias = parts[9]

        v_final = None
        noneVs = ['NULL', '', '-', '其它', '其他', '无']
        if(v not in noneVs):
          v_final = v
        elif (v in noneVs and v_rem not in noneVs):
          v_final = v_rem
        elif (v in noneVs and v_alias not in noneVs):
          v_final = v_alias
        else:
          errCnt += 1
          continue
        w.write(sku + '\t' + r1 + '_' + r2 + '\t' + v_final + '\n')
  print('Filtered {} line'.format(errCnt))

def filter_kb_ext_ori(input_file):
  with open(input_file, 'r', encoding='utf-8') as f:
    with open(input_file+'.filtered', 'w', encoding='utf-8') as w:
      errCnt = 0
      while(True):
        line = f.readline()
        if not line:
          break
        line = line.strip()
        parts = line.split('\t')
        sku = parts[0]
        r = parts[2]
        v = parts[4]
        v_rem = ''
        v_unit = ''
        v_alias = ''

        v_final = None
        noneVs = ['NULL', '', '-', '其它', '其他']
        if(v not in noneVs):
          v_final = v
        elif(v in noneVs and v_rem not in noneVs):
          v_final = v_rem
        else:
          errCnt += 1
          continue
        w.write(sku + '\t' + r + '\t' + v_final + '\n')
  print('Filtered {} line'.format(errCnt))

def filter_kb_ext(input_file):
  with open(input_file, 'r', encoding='utf-8') as f:
    with open(input_file+'.filtered', 'w', encoding='utf-8') as w:
      errCnt = 0
      while(True):
        line = f.readline()
        if not line:
          break
        line = line.strip()
        parts = line.split('\t')
        sku = parts[0]
        r = parts[2]
        v = parts[4]
        v_rem = parts[5]
        v_unit = parts[6]
        v_alias = parts[7]

        v_final = None
        noneVs = ['NULL', '', '-', '其它', '其他']
        if(v not in noneVs):
          v_final = v
        elif(v in noneVs and v_rem not in noneVs):
          v_final = v_rem
        else:
          errCnt += 1
          continue
        w.write(sku + '\t' + r + '\t' + v_final + '\n')
  print('Filtered {} line'.format(errCnt))

def print_v_Null_data_spec(input_file):
  with open(input_file, 'r', encoding='utf-8') as f:
    with open(input_file+'.valueNull', 'w', encoding='utf-8') as w:
      while(True):
        line = f.readline()
        if not line:
          break
        line = line.strip('\n')
        parts = line.split('\t')
        if(len(parts) != 10):
          print(line)
        assert len(parts) == 10
        sku = parts[0]
        r1 = parts[2]
        r2 = parts[4]
        v = parts[6]
        v_rem = parts[7]
        v_unit = parts[8]
        v_alias = parts[9]

        # if (v == 'NULL' or v == '其它' or v == '其他' or v_rem != 'NULL' or v_unit != 'NULL' or v_alias != 'NULL'):
        if (v_rem != 'NULL' or v_unit != 'NULL' or v_alias != 'NULL'):
          w.write(line + '\n')

def print_v_Null_data_ext(input_file):
  with open(input_file, 'r', encoding='utf-8') as f:
    with open(input_file+'.valueNull', 'w', encoding='utf-8') as w:
      while(True):
        line = f.readline()
        if not line:
          break
        line = line.strip()
        parts = line.split('\t')
        sku = parts[0]
        r1 = parts[2]
        v = parts[4]
        v_rem = parts[5]
        v_unit = parts[6]
        v_alias = parts[7]

        # if (v == 'NULL' or v == '其它' or v == '其他' or v_rem != 'NULL' or v_unit != 'NULL' or v_alias != 'NULL'):
        noneVs = ['NULL', '', '-']
        if (v_rem not in noneVs or v_unit not in noneVs or v_alias not in noneVs):

          w.write(line + '\n')

def load_kb(file, id2kv):
  # file = r"/home/baojunwei/notespace/tensorflow-models/data_JD/kb/jd.kb"
  with open(file, mode='r', encoding='utf-8') as f_in:
    cnt = 0
    for line in f_in:
      parts = line.strip('\n ').split('\t')
      sbj = parts[0]
      pred = parts[1]
      obj = parts[2]

      if sbj not in id2kv:
        id2kv[sbj] = []
        cnt += 1
      id2kv[sbj].append((pred, obj))
      if (cnt % 1000 == 0):
        print("Loading %d lines.\r" % cnt)
  print("KB Length: %d\n" % len(id2kv))
  return id2kv

def load_partial_kb(file, sku_list, id2kv):
  # file = r"/home/baojunwei/notespace/tensorflow-models/data_JD/kb/jd.kb"
  with open(file, mode='r', encoding='utf-8') as f_in:
    cnt = 0
    for line in f_in:
      parts = line.strip('\n ').split('\t')
      sbj = parts[0]
      if sbj not in sku_list:
        continue
      pred = parts[1]
      obj = parts[2]

      if sbj not in id2kv:
        id2kv[sbj] = []
        cnt += 1
      id2kv[sbj].append((pred, obj))
      if (cnt % 100 == 0):
        print("Loading %d lines.\r" % cnt)
  print("KB Length: %d\n" % len(id2kv))
  return id2kv

def join_writing_kb(file, kb):
  f_out = open(file + ".joinKB", 'w', encoding='utf-8')
  with open(file, 'r', encoding='utf-8') as f_in:
    index = 0.0
    cnt = 0.0
    for line in f_in:
      parts = line.split('\t')
      sku = parts[0]
      sku_name = parts[1]
      brand_code = parts[2]
      brandname_en = parts[3]
      brandname_cn = parts[4]
      brandname_full = parts[5]
      catId1 = parts[6]
      catName1 = parts[7]
      catId2 = parts[8]
      catName2 = parts[9]
      catId3 = parts[10]
      catName3 = parts[11]
      writingId = parts[12]
      picture = parts[13]
      labels = parts[14]
      title = parts[15]
      writing = parts[16].strip('\n')

      kvs = kb.get(sku, [])
      if(len(kvs)!= 0):
        index += 1
        f_out.write("AlphaWriting-%d\n" %index)
        f_out.write("WritingID:\t%s\n" %writingId)
        f_out.write("Title:\t%s\n" %title)
        f_out.write("Text:\t%s\n" %writing)
        f_out.write("Picture:\t%s\n" %picture)
        f_out.write("Sku:\t%s\n" %sku)
        f_out.write("KB:\tSkuName ||| %s\n" %sku_name)
        f_out.write("KB:\tBrandName_full ||| %s\n" %(brandname_full))
        f_out.write("KB:\tBrandName_cn ||| %s\n" %brandname_cn)
        f_out.write("KB:\tBrandName_en ||| %s\n" %brandname_en)
        f_out.write("KB:\tLabels ||| %s\n" % labels)
        f_out.write("KB:\tCategoryL1 ||| %s\n" %(catName1))
        f_out.write("KB:\tCategoryL2 ||| %s\n" %(catName2))
        f_out.write("KB:\tCategoryL3 ||| %s\n" %(catName3))
        for k, v in kvs:
          f_out.write("KB:\t%s ||| %s\n"%(k,v))
        f_out.write("\n")
  f_out.close()

def reduce_skuName_label_in_writing(file_in):
  dict = {}  # key: sku + writingId
  with open(file_in, 'r', encoding='utf-8') as f_in:
    for line in f_in:
      parts = line.strip('\n').split('\t')
      sku = parts[0]
      sku_name = parts[1]
      brand_code = parts[2]
      brandname_en = parts[3]
      brandname_cn = parts[4]
      brandname_full = parts[5]
      catId1 = parts[6]
      catName1 = parts[7]
      catId2 = parts[8]
      catName2 = parts[9]
      catId3 = parts[10]
      catName3 = parts[11]
      sku_dt = parts[12]
      contentId = parts[13]
      picture = parts[14]
      labels = parts[15].split(',')
      title = parts[16]
      description = parts[17]

      key = sku + "\t" + contentId
      item = dict.get(key, None)
      if(not item):
        item = (sku, [sku_name], brand_code, brandname_en, brandname_cn, brandname_full, catId1, catName1, catId2, catName2, catId3, catName3, contentId, [picture], labels, [title], description)
      else:
        item[1].append(sku_name)
        item[13].append(picture)
        item[14].extend(labels)
        item[15].append(title)

      dict[key] = item

  with open(file_in + ".reduced", 'w', encoding='utf-8') as f_out:
    for key, item in dict.items():
      f_out.write(item[0] + "\t")#sku
      f_out.write(item[1][0]+ "\t")#sku_name
      f_out.write(item[2] + "\t")  # brand_code
      f_out.write(item[3] + "\t")  # brandname_en
      f_out.write(item[4] + "\t")  # brandname_cn
      f_out.write(item[5] + "\t")  # brandname_full
      f_out.write(item[6]+ "\t")#catId1
      f_out.write(item[7]+ "\t")#catName1
      f_out.write(item[8]+ "\t")#catId2
      f_out.write(item[9]+ "\t")#catName2
      f_out.write(item[10]+ "\t")#catId3
      f_out.write(item[11]+ "\t")#catName3
      f_out.write(item[12]+ "\t")#contentId
      f_out.write(item[13][0]+ "\t")#picture
      f_out.write(",".join(set(item[14]))+ "\t")#labels
      f_out.write(item[15][0]+ "\t")#title
      f_out.write(item[16]+ "\n")#description

def load_sku_basic_info(sku_basic_info_file, sku_list):
  sku2basicInfo = {}
  with open(sku_basic_info_file, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      parts = line.split('\t')
      Sku = parts[0]
      if Sku not in sku_list:
        continue
      SkuName = parts[1]
      Brand_id = parts[2]
      BrandName_en = parts[3]
      BrandName_cn = parts[4]
      BrandName_full = parts[5]
      CategoryL1_id = parts[6]
      CategoryL1 = parts[7]
      CategoryL2_id = parts[8]
      CategoryL2 = parts[9]
      CategoryL3_id = parts[10]
      CategoryL3 = parts[11]
      sku2basicInfo[Sku]={'Sku': Sku,
                          'SkuName': SkuName,
                          'Brand_id': Brand_id,
                          'BrandName_en': BrandName_en,
                          'BrandName_cn': BrandName_cn,
                          'BrandName_full': BrandName_full,
                          'CategoryL1_id': CategoryL1_id,
                          'CategoryL1': CategoryL1,
                          'CategoryL2_id': CategoryL2_id,
                          'CategoryL2': CategoryL2,
                          'CategoryL3_id': CategoryL3_id,
                          'CategoryL3': CategoryL3}
    return sku2basicInfo

def load_sku2ocr(sku_ocr_file):
  sku2ocr = {}
  with open(sku_ocr_file, 'r', encoding='utf-8') as f:
    while(True):
      line = f.readline()
      if not line:
        break
      line = line.strip('\n')
      parts = line.split('\t')
      if len(parts) != 4:
        print(line)
      sku = parts[0]
      ocr = parts[4]
      if sku not in sku2ocr:
        sku2ocr[sku] = []
      if ocr != '':
        sku2ocr[sku].append(ocr)
  return sku2ocr

def get_input_info_for_new_sku(new_sku_file, kb, skuBasicInfo, validCat3list):
  with open(new_sku_file + '.basicInfo', 'w', encoding='utf-8') as f_out_basicInfo:
    f_out_basicInfo.write('\t'.join(['Sku',
                          'SkuName',
                          'Brand_id',
                          'BrandName_en',
                          'BrandName_cn',
                          'BrandName_full',
                          'CategoryL1_id',
                          'CategoryL1',
                          'CategoryL2_id',
                          'CategoryL2',
                          'CategoryL3_id',
                          'CategoryL3']) + '\n')
    with open(new_sku_file + '.table.forDebug', 'w', encoding='utf-8') as f_out_table_debug:
      with open(new_sku_file + '.table', 'w', encoding='utf-8') as f_out_table:
        with open(new_sku_file + '.fakeWriting', 'w', encoding='utf-8') as f_out_writing:
          with open(new_sku_file + '.validSku', 'w', encoding='utf-8') as f_out_sku:
            with open(new_sku_file + '.validSku.flag', 'w', encoding='utf-8') as f_out_flag:
              with open(new_sku_file, 'r', encoding='utf-8') as f:
                skus = [_.strip().replace('.html','').replace(r'http://item.jd.com/','') for _ in f]
                index = 0
                for sku in skus:
                  if sku in kb and sku in skuBasicInfo and skuBasicInfo[sku]['CategoryL3'] in validCat3list:
                    kvs = kb.get(sku, [])
                    if (len(kvs) != 0):
                      index += 1
                      f_out_basicInfo.write('\t'.join(skuBasicInfo[sku].values()) + '\n')
                      all_kvs = []
                      all_kvs.append(('SkuName',skuBasicInfo[sku]['SkuName']))
                      all_kvs.append(('BrandName_full',skuBasicInfo[sku]['BrandName_full']))
                      all_kvs.append(('BrandName_cn',skuBasicInfo[sku]['BrandName_cn']))
                      all_kvs.append(('BrandName_en',skuBasicInfo[sku]['BrandName_en']))
                      all_kvs.append(('CategoryL1',skuBasicInfo[sku]['CategoryL1']))
                      all_kvs.append(('CategoryL2',skuBasicInfo[sku]['CategoryL2']))
                      all_kvs.append(('CategoryL3',skuBasicInfo[sku]['CategoryL3']))
                      for k, v in kvs:
                        all_kvs.append((k, v))

                      f_out_table_debug.write("Index:\t%d\n" % index)
                      f_out_table_debug.write("Sku:\t%s\n" % sku)
                      f_out_table_debug.write("Brand_id:\t%s\n" % skuBasicInfo[sku]['Brand_id'])
                      f_out_table_debug.write("CategoryL1_id:\t%s\n" % skuBasicInfo[sku]['CategoryL1_id'])
                      f_out_table_debug.write("CategoryL2_id:\t%s\n" % skuBasicInfo[sku]['CategoryL2_id'])
                      f_out_table_debug.write("CategoryL3_id:\t%s\n" % skuBasicInfo[sku]['CategoryL3_id'])
                      for k, v in all_kvs:
                        f_out_table_debug.write("KB:\t%s ||| %s\n" % (k, v))
                      f_out_table_debug.write("\n")

                      f_out_table.write(' ## '.join([k.strip() + ' $$ ' + v.strip() for k, v in all_kvs]) + "\n")
                      f_out_sku.write(sku + '\n')
                      f_out_flag.write('1\n')
                      f_out_writing.write('Fake Writing!\n')

def get_input_info_and_filter_writing_for_new_sku(new_sku_file, kb, skuBasicInfo, writing_file):
  with open(new_sku_file + '.basicInfo', 'w', encoding='utf-8') as f_out_basicInfo:
    f_out_basicInfo.write('\t'.join(['Sku',
                          'SkuName',
                          'Brand_id',
                          'BrandName_en',
                          'BrandName_cn',
                          'BrandName_full',
                          'CategoryL1_id',
                          'CategoryL1',
                          'CategoryL2_id',
                          'CategoryL2',
                          'CategoryL3_id',
                          'CategoryL3']) + '\n')
    with open(new_sku_file + '.validSkus', 'w', encoding='utf-8') as f_out_validSkus:
      with open(new_sku_file + '.table.forDebug', 'w', encoding='utf-8') as f_out_table_debug:
        with open(new_sku_file + '.table', 'w', encoding='utf-8') as f_out_table:
          with open(new_sku_file + '.validWriting', 'w', encoding='utf-8') as f_out_writing:
            with open(writing_file, 'r', encoding='utf-8') as f:
              writings = [_.strip() for _ in f.readlines()]
            with open(new_sku_file, 'r', encoding='utf-8') as f:
              skus = [_.strip().replace('.html','').replace(r'http://item.jd.com/','') for _ in f]
              id = 0
              index = 0
              for sku in skus:
                writing = writings[id]
                if sku in kb and sku in skuBasicInfo:
                  kvs = kb.get(sku, [])
                  if (len(kvs) != 0):
                    index += 1
                    f_out_basicInfo.write('\t'.join(skuBasicInfo[sku].values()) + '\n')
                    all_kvs = []
                    all_kvs.append(('SkuName',skuBasicInfo[sku]['SkuName']))
                    all_kvs.append(('BrandName_full',skuBasicInfo[sku]['BrandName_full']))
                    all_kvs.append(('BrandName_cn',skuBasicInfo[sku]['BrandName_cn']))
                    all_kvs.append(('BrandName_en',skuBasicInfo[sku]['BrandName_en']))
                    all_kvs.append(('CategoryL1',skuBasicInfo[sku]['CategoryL1']))
                    all_kvs.append(('CategoryL2',skuBasicInfo[sku]['CategoryL2']))
                    all_kvs.append(('CategoryL3',skuBasicInfo[sku]['CategoryL3']))
                    for k, v in kvs:
                      all_kvs.append((k, v))

                    f_out_table_debug.write("Index:\t%d\n" % index)
                    f_out_table_debug.write("Sku:\t%s\n" % sku)
                    f_out_table_debug.write("Brand_id:\t%s\n" % skuBasicInfo[sku]['Brand_id'])
                    f_out_table_debug.write("CategoryL1_id:\t%s\n" % skuBasicInfo[sku]['CategoryL1_id'])
                    f_out_table_debug.write("CategoryL2_id:\t%s\n" % skuBasicInfo[sku]['CategoryL2_id'])
                    f_out_table_debug.write("CategoryL3_id:\t%s\n" % skuBasicInfo[sku]['CategoryL3_id'])
                    for k, v in all_kvs:
                      f_out_table_debug.write("KB:\t%s ||| %s\n" % (k, v))
                    f_out_table_debug.write("\n")

                    f_out_table.write(' ## '.join([k.strip() + ' $$ ' + v.strip() for k, v in all_kvs]) + "\n")

                    f_out_writing.write(writing + '\n')
                    f_out_validSkus.write(sku + '\n')
                id += 1

def dumpOCR(sku_file, ocr_file):
  return

if __name__ == "__main__":

  if(False): #data_processing_step-1.1: KB Processing
    # print_v_Null_data_ext(r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData_0609\jiayongdianqi.kb.ext.0610')
    # filter_kb_ext_ori(r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData_0622\jiayongdianqi.kb.ext.0622')
    filter_kb_ext(r'/export/homes/baojunwei/alpha-w-pipeline/data/chuju_rawData/chuju.kb.ext')

    # print_v_Null_data_spec(r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData_0609\jiayongdianqi.kb.spec.0609')
    # filter_kb_spec(r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData_0622\jiayongdianqi.kb.spec.0622')
    filter_kb_spec(r'/export/homes/baojunwei/alpha-w-pipeline/data/chuju_rawData/chuju.kb.spec')

  if(False): #data_processing_step-1.3: join writing with KB [For expriments: train/dev/test]
    sku_ext_info_file = r'/export/homes/baojunwei/alpha-w-pipeline/data/chuju_rawData/chuju.kb.ext.filtered'
    sku_spec_info_file = r'/export/homes/baojunwei/alpha-w-pipeline/data/chuju_rawData/chuju.kb.spec.filtered.concat'
    sku_writing_file = r'/export/homes/baojunwei/alpha-w-pipeline/data/chuju_rawData/chuju.writing'
    # sku_ext_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.ext.0408.filtered'
    # sku_spec_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.spec.0408.filtered.concat'
    # sku_writing_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.writing'
    # sku_ext_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData_0609\jiayongdianqi.kb.ext.0610.filtered'
    # sku_spec_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData_0609\jiayongdianqi.kb.spec.0609.filtered.concat'
    # sku_writing_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData_0609\jiayongdianqi.writing.0609'
    reduce_skuName_label_in_writing(sku_writing_file)  # jiayongdianqi.writing
    kb = {}
    kb = load_kb(sku_ext_info_file, kb)  # jiayongdianqi.kb.ext.0408.filtered
    kb = load_kb(sku_spec_info_file, kb)  # jiayongdianqi.kb.spec.0408.filtered.concat
    join_writing_kb(sku_writing_file + '.reduced', kb)  # jiayongdianqi.writing.reduced

  if(False): #xuanpin1: data_processing_step-1: prepare sku infomation including sku title and KB [For prediction]
    # sku_basic_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.skuInfo'
    # sku_ext_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.ext.0408.filtered'
    # sku_spec_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.spec.0408.filtered.concat'
    # test_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin/xuanpin.merge.sku'
    sku_basic_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.skuInfo'
    sku_ext_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.ext.0408.filtered'
    sku_spec_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.spec.0408.filtered.concat'
    test_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin1\xuanpin.merge.sku'

    skuBasicInfo = load_sku_basic_info(sku_basic_info_file)
    kb = {}
    kb = load_kb(sku_ext_info_file, kb)  # jiayongdianqi.kb.ext.0408.filtered
    kb = load_kb(sku_spec_info_file, kb)  # jiayongdianqi.kb.spec.0408.filtered.concat
    get_input_info_for_new_sku(test_file, kb, skuBasicInfo)

  if(False): #xuanpin2: data_processing_step-1: prepare sku infomation including sku title and KB [For prediction]
    # sku_basic_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.skuInfo'
    # sku_ext_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.ext.0408.filtered'
    # sku_spec_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.spec.0408.filtered.concat'
    # test_file = r'    test_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin_2/xuanpin2.sku'
    sku_basic_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.skuInfo'
    sku_ext_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.ext.0408.filtered'
    sku_spec_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.spec.0408.filtered.concat'
    test_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin2\xuanpin2.sku'

    skuBasicInfo = load_sku_basic_info(sku_basic_info_file)
    kb = {}
    kb = load_kb(sku_ext_info_file, kb)  # jiayongdianqi.kb.ext.0408.filtered
    kb = load_kb(sku_spec_info_file, kb)  # jiayongdianqi.kb.spec.0408.filtered.concat
    get_input_info_for_new_sku(test_file, kb, skuBasicInfo)

  if(False): #xuanpin3: data_processing_step-1: prepare sku infomation including sku title and KB [For prediction]
    # sku_basic_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.skuInfo'
    # sku_ext_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.ext.0408.filtered'
    # sku_spec_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.spec.0408.filtered.concat'
    # test_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin2/xuanpin2'
    sku_basic_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.skuInfo'
    sku_ext_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.ext.0408.filtered'
    sku_spec_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.spec.0408.filtered.concat'
    test_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin3\xuanpin3'

    skuBasicInfo = load_sku_basic_info(sku_basic_info_file)
    kb = {}
    kb = load_kb(sku_ext_info_file, kb)  # jiayongdianqi.kb.ext.0408.filtered
    kb = load_kb(sku_spec_info_file, kb)  # jiayongdianqi.kb.spec.0408.filtered.concat
    get_input_info_for_new_sku(test_file, kb, skuBasicInfo)

  if(False): #xuanpin4: data_processing_step-1: prepare sku infomation including sku title and KB [For prediction]
    # sku_basic_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.skuInfo'
    # sku_ext_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.ext.0408.filtered'
    # sku_spec_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.spec.0408.filtered.concat'
    # test_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin2/xuanpin2'

    # sku_basic_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.skuInfo'
    # sku_ext_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.ext.0408.filtered'
    # sku_spec_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.spec.0408.filtered.concat'
    # test_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\xuanpin4'
    #
    # skuBasicInfo = load_sku_basic_info(sku_basic_info_file)
    # kb = {}
    # kb = load_kb(sku_ext_info_file, kb)  # jiayongdianqi.kb.ext.0408.filtered
    # kb = load_kb(sku_spec_info_file, kb)  # jiayongdianqi.kb.spec.0408.filtered.concat
    # get_input_info_for_new_sku(test_file, kb, skuBasicInfo)

    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\xuanpin4.validSku'
    ocr_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\xuanpin4.ocr'
    valid_ocr_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin4\xuanpin4.validOcr'

    sku2ocr = load_sku2ocr(ocr_file)
    valid_skus = [_.strip() for _ in open(valid_sku_file, 'r', encoding='utf-8')]
    with open(valid_ocr_file, 'w', encoding='utf-8') as f:
      for sku in valid_skus:
        if sku in sku2ocr:
          f.write('\t'.join(sku2ocr[sku]) + '\n')
        else:
          f.write('\n')

  if(False): #xuanpin6: data_processing_step-1: prepare sku infomation including sku title and KB [For prediction]
    # sku_basic_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.skuInfo'
    # sku_ext_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.ext.0408.filtered'
    # sku_spec_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.spec.0408.filtered.concat'

    sku_basic_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.skuInfo'
    sku_ext_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.ext.0408.filtered'
    sku_spec_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.spec.0408.filtered.concat'
    sku_basic_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData_0622\jiayongdianqi.kb.skuInfo.0622'
    sku_ext_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData_0622\jiayongdianqi.kb.ext.0622.filtered'
    sku_spec_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData_0622\jiayongdianqi.kb.spec.0622.filtered.concat'

    test_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin6\xuanpin6'
    sku_list = [_.strip() for _ in open(test_file, 'r', encoding='utf-8')]

    skuBasicInfo = load_sku_basic_info(sku_basic_info_file, sku_list)
    kb = {}
    kb = load_partial_kb(sku_ext_info_file, sku_list, kb)  # jiayongdianqi.kb.ext.0408.filtered
    kb = load_partial_kb(sku_spec_info_file, sku_list, kb)  # jiayongdianqi.kb.spec.0408.filtered.concat
    get_input_info_for_new_sku(test_file, kb, skuBasicInfo)

    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin6\xuanpin6.validSku'
    ocr_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin6\xuanpin6.ocr'
    valid_ocr_file = r'D:\Research\Projects\AlphaWriting\model\data\xuanpin6\xuanpin6.validOcr'

    sku2ocr = load_sku2ocr(ocr_file)
    valid_skus = [_.strip() for _ in open(valid_sku_file, 'r', encoding='utf-8')]
    with open(valid_ocr_file, 'w', encoding='utf-8') as f:
      for sku in valid_skus:
        if sku in sku2ocr:
          f.write('\t'.join(sku2ocr[sku]) + '\n')
        else:
          f.write('\n')

  if(False): #alphaSales: data_processing_step-1: prepare sku infomation including sku title and KB [For prediction]
    # sku_basic_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.skuInfo'
    # sku_ext_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.ext.0408.filtered'
    # sku_spec_info_file = r'/home/baojunwei/project/alpha_writing/data/jiayongdianqi_rawData/jiayongdianqi.kb.spec.0408.filtered.concat'
    # test_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin2/xuanpin2'
    # sku_basic_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.skuInfo'
    # sku_ext_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.ext.0408.filtered'
    # sku_spec_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.spec.0408.filtered.concat'
    sku_basic_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData_0622\jiayongdianqi.kb.skuInfo.0622'
    sku_ext_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData_0622\jiayongdianqi.kb.ext.0622.filtered'
    sku_spec_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData_0622\jiayongdianqi.kb.spec.0622.filtered.concat'

    test_file = r'D:\Research\Projects\AlphaWriting\model\data\alphaSales\alphaSales'
    sku_list = [_.strip() for _ in open(test_file, 'r', encoding='utf-8')]
    skuBasicInfo = load_sku_basic_info(sku_basic_info_file, sku_list)
    kb = {}
    kb = load_partial_kb(sku_ext_info_file, sku_list, kb)  # jiayongdianqi.kb.ext.0408.filtered
    kb = load_partial_kb(sku_spec_info_file, sku_list, kb)  # jiayongdianqi.kb.spec.0408.filtered.concat
    get_input_info_for_new_sku(test_file, kb, skuBasicInfo)

    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\alphaSales\alphaSales.validSku'
    ocr_file = r'D:\Research\Projects\AlphaWriting\model\data\alphaSales\alphaSales.ocr'
    valid_ocr_file = r'D:\Research\Projects\AlphaWriting\model\data\alphaSales\alphaSales.validOcr'

    sku2ocr = load_sku2ocr(ocr_file)
    valid_skus = [_.strip() for _ in open(valid_sku_file, 'r', encoding='utf-8')]
    with open(valid_ocr_file, 'w', encoding='utf-8') as f:
      for sku in valid_skus:
        if sku in sku2ocr:
          f.write('\t'.join(sku2ocr[sku]) + '\n')
        else:
          f.write('\n')

  if(False): #chuju-test: data_processing_step-1: prepare sku infomation including sku title and KB [For prediction]
    root = r'/export/homes/baojunwei/alpha-w-pipeline/data'
    dataset = 'test'
    sku_basic_info_file = os.path.join(root, r'chuju_rawData/chuju.kb.skuInfo')
    sku_ext_info_file = os.path.join(root, r'chuju_rawData/chuju.kb.ext.filtered')
    sku_spec_info_file = os.path.join(root, r'chuju_rawData/chuju.kb.spec.filtered.concat')
    # test_file = r'/home/baojunwei/project/alpha_writing/data/xuanpin2/xuanpin2'
    # sku_basic_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.skuInfo'
    # sku_ext_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.ext.0408.filtered'
    # sku_spec_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.spec.0408.filtered.concat'
    # sku_basic_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData_0622\jiayongdianqi.kb.skuInfo.0622'
    # sku_ext_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData_0622\jiayongdianqi.kb.ext.0622.filtered'
    # sku_spec_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData_0622\jiayongdianqi.kb.spec.0622.filtered.concat'

    # test_file = r'D:\Research\Projects\AlphaWriting\model\data\alphaSales\alphaSales'
    test_file = os.path.join(root, dataset, dataset)
    sku_list = [_.strip() for _ in open(test_file, 'r', encoding='utf-8')]
    skuBasicInfo = load_sku_basic_info(sku_basic_info_file, sku_list)
    kb = {}
    kb = load_kb(sku_ext_info_file, sku_list, kb)  # jiayongdianqi.kb.ext.0408.filtered
    kb = load_kb(sku_spec_info_file, sku_list, kb)  # jiayongdianqi.kb.spec.0408.filtered.concat
    get_input_info_for_new_sku(test_file, kb, skuBasicInfo, ['茶壶', '保温杯', '玻璃杯', '茶杯'])

  if(True): #chuju-test: data_processing_step-1: prepare sku infomation including sku title and KB [For prediction]
    #root = r'/export/homes/baojunwei/alpha-w-pipeline/data'
    root = os.path.join(sys.argv[1], 'data')
    domain = sys.argv[2]#'chuju'
    dataset = sys.argv[3]#'test'
    ocr_file = os.path.join(root, domain + "_" + dataset, dataset + r'.ocr')
    valid_sku_file = os.path.join(root, domain + "_" + dataset, dataset + r'.validSku')
    valid_ocr_file = os.path.join(root, domain + "_" + dataset, dataset + r'.validOcr')

    sku2ocr = load_sku2ocr(ocr_file)
    valid_skus = [_.strip() for _ in open(valid_sku_file, 'r', encoding='utf-8')]
    with open(valid_ocr_file, 'w', encoding='utf-8') as f:
      for sku in valid_skus:
        if sku in sku2ocr:
          f.write('\t'.join(sku2ocr[sku]) + '\n')
        else:
          f.write('\n')

  if(False): #dump validOcr for encoding
    dataset = 'dev'
    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\filtered_data_v2\\'+dataset+'.sku'
    ocr_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.ocr'
    valid_ocr_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\filtered_data_v2\\'+dataset+'.validOcr'

    sku2ocr = load_sku2ocr(ocr_file)
    valid_skus = [_.strip() for _ in open(valid_sku_file, 'r', encoding='utf-8')]
    with open(valid_ocr_file, 'w', encoding='utf-8') as f:
      for sku in valid_skus:
        if sku in sku2ocr:
          f.write('\t'.join(sku2ocr[sku]) + '\n')
        else:
          f.write('\n')

  if(False): #dump validOcr for decoding
    dataset = 'dev'
    valid_sku_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\filtered_data_v2\\'+dataset+'.sku'
    ocr_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.ocr'
    valid_ocr_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi\filtered_data_v2\\'+dataset+'.validOcr'

    sku2ocr = load_sku2ocr(ocr_file)
    valid_skus = [_.strip() for _ in open(valid_sku_file, 'r', encoding='utf-8')]
    with open(valid_ocr_file, 'w', encoding='utf-8') as f:
      for sku in valid_skus:
        if sku in sku2ocr:
          f.write('\t'.join(sku2ocr[sku]) + '\n')
        else:
          f.write('\n')

  if(False): #review: data_processing_step-1: prepare sku infomation including sku title and KB [For reviewing]
    sku_basic_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.skuInfo'
    sku_ext_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.ext.0408.filtered'
    sku_spec_info_file = r'D:\Research\Projects\AlphaWriting\model\data\jiayongdianqi_rawData\jiayongdianqi.kb.spec.0408.filtered.concat'
    test_file = r'D:\Research\Projects\AlphaWriting\model\data\writingReview\review.sku'
    writing_file = r'D:\Research\Projects\AlphaWriting\model\data\writingReview\review.writing'

    skuBasicInfo = load_sku_basic_info(sku_basic_info_file)
    kb = {}
    kb = load_kb(sku_ext_info_file, kb)  # jiayongdianqi.kb.ext.0408.filtered
    kb = load_kb(sku_spec_info_file, kb)  # jiayongdianqi.kb.spec.0408.filtered.concat
    get_input_info_and_filter_writing_for_new_sku(test_file, kb, skuBasicInfo, writing_file)



