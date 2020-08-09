#!/usr/bin/bash
set -e
<<comment
#training phase
python fast_stable_texture_transfer_feedforward.py \
        --training 1 \
	--style_image ./data/mosaic.jpg \
	--style_scale 1 \
        --log_interval 500 \
        --checkpoint_interval 2000 \
	--style_weight 1e2 \
	--content_weight 1e0 \
	--image_size 256 \
	--image_style_size 256 \
        --save_model_dir models/saved/mosaic/ \
        --checkpoint_model_dir models/checkpoints/mosaic/ \
	--epoch 1 \
        --learning_rate 1e-4 \
        --dataset /media/james/alibaba2/FILES_OFFICE/Researches/Style-transfer/Fast-neural-style-transfer/images/coco 
comment
#<<comment
#testing phase
python fast_stable_texture_transfer_feedforward.py \
	--training 0 \
	--content_image ./data/44.png \
	--image_size 512 \
	--style_model models/checkpoints/mosaic/ckpt_epoch_1_batch_id_2000.pth \
	--output_image ./data/44-2-mosaic.png \
	--original_colors 0

#comment
