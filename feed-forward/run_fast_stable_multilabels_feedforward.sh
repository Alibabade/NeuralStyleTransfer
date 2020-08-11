#!/usr/bin/bash
set -e

#,./data/mosaic_mask2.png,./data/5_mask3.png \
#,./data/mosaic.jpg,./data/5.jpg \
<<comment
#run training process
python fast_stable_texture_transfer_multilabels_feedforward.py \
        --training 1 \
        --style_images ./data/7.jpg \
	--content_mask_image ./data/44_mask5.png \
	--style_mask_images ./data/butterfly_mask4.png \
	--style_scale 1 \
	--semantic 0 \
        --learning_rate 1e-4 \
        --content_weight 1e0 \
        --log_interval 500 \
        --checkpoint_interval 2000 \
        --save_model_dir models/saved/7/ \
        --checkpoint_model_dir models/checkpoints/7/ \
        --epoch 2 \
        --dataset /media/james/alibaba2/FILES_OFFICE/Researches/Style-transfer/Fast-neural-style-transfer/images/coco
comment
#<<comment
#run testing process
python fast_stable_texture_transfer_multilabels_feedforward.py \
        --training 0 \
        --semantic 0 \
        --original_colors 0 \
        --image_size 846 \
        --content_image ./data/674.jpg \
        --content_mask_image ./data/44_mask5.png \
        --style_mask_images ./data/butterfly_mask4.png,./data/mosaic_mask2.png,./data/5_mask3.png \
        --output_image ./output/674_6_stylized.png \
        --style_model models/saved/6/epoch_2_Tue_Aug_11_07:57:58_2020_content_1.0_style_100.0_semantic_0.model
#comment

