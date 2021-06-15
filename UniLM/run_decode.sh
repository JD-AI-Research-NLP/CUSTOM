# STEP=28
# MAXLEN=(20 30 50 60 70 90 100 110 120 130 140 150 160 170 180 190 200)
# # MAXLEN=(128)
# for len in ${MAXLEN[@]}
# do
#     echo "decode length ${len}"
#     python3 -u decode_seq2seq.py --model_type unilm --model_name_or_path ./pretrained_model/ --model_recover_path ./finetuned_model/shouji/model.${STEP}.bin --max_seq_length 512 --input_file ./data/shouji/test.input --input_aspect ./data/shouji/test.aspect --output_file ./predict_results/shouji/predict.${STEP}.txt_tgt${len} --do_lower_case --batch_size 32 --beam_size 5 --max_tgt_length ${len}
# done

# STEP=30
# # MAXLEN=(20 30 50 60 70 90 100 110 120 130 140 150 160 170 180 190 200)
# MAXLEN=(50 60 70 80 90 100)
# # MAXLEN=(80)
# for len in ${MAXLEN[@]}
# do
#     echo "decode length ${len}"
#     python3 -u decode_seq2seq.py --model_type unilm --model_name_or_path ./pretrained_model/ --model_recover_path ./finetuned_model/shouji/model.${STEP}.bin --max_seq_length 512 --input_file ./data/shouji/test.input --input_aspect ./data/shouji/test.aspect --output_file ./predict_results/shouji/predict.${STEP}.txt_tgt${len} --do_lower_case --batch_size 32 --beam_size 5 --max_tgt_length ${len} --fp16
# done 

# STEP=(17 18 19 20 21 22 23 24 25 26 27 28 29 30)
STEP=(25 30)
MAXLEN=80
for step in ${STEP[@]}
do
    echo "decode step ${step}"
    python3 -u decode_seq2seq.py --model_type unilm --model_name_or_path ./pretrained_model/ --model_recover_path ./finetuned_model/shouji/model.${step}.bin --max_seq_length 512 --input_file ./data/shouji/test.input --input_aspect ./data/shouji/test.aspect --output_file ./predict_results/shouji/predict.${step}.txt_tgt${MAXLEN} --do_lower_case --batch_size 32 --beam_size 5 --max_tgt_length ${MAXLEN} 
done