# Eval
DATA_PATH=[Your MSRVTT data and videos path]
python -m torch.distributed.launch --nproc_per_node=4 \
main_task_retrieval.py --do_eval --num_thread_reader=0 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSRVTT_data.json \
--features_path ${DATA_PATH}/MSRVTT_Videos \
--init_model /path/to/your_checkpoint \
--output_dir ckpts/ckpt_msrvtt_eval \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt --expand_msrvtt_sentences --seed 42 \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32 \
--use_moe --vis_top_k 1 \
--visual_moe_indices 10 11 \
--text_moe_indices 10 11 \
--video_frames 12 --ctia_alpha "1.0,0.5,0.5" \
--use_load_balancing_loss --add_cls_num 2 \
--use_hard_negative --hard_neg_margin "1.0" 

