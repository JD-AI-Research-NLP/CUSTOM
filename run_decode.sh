STEP=30
MODEL_PATH="./finetuned_model/shouji"
MAXLEN=80
python3 -u decode_seq2seq.py --model_type unilm --model_name_or_path ./pretrained_model/ --model_recover_path MODEL_PATH/model.${step}.bin --max_seq_length 512 --input_file ./data/shouji/test.input --input_aspect ./data/shouji/test.aspect --output_file ./predict_results/shouji/predict.${step}.txt_tgt${MAXLEN} --do_lower_case --batch_size 32 --beam_size 5 --max_tgt_length ${MAXLEN} 