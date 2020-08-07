
python fast_stable_texture_transfer_multilabels.py \
        --content_image ./data/41.png \
        --style_image ./data/butterfly.jpg \
	--content_mask_image ./data/41_mask2.png \
	--style_mask_image ./data/butterfly_mask.png \
	--style_scale 1 \
	--semantic 1 \
        --output_image ./output/41-2-butterfly_opt_multilabels.png \
        --max_nums 1000 \
        --original_colors 1 \
        --content_layers ['4','9','16','23'] \
        --style_weight 1e8 \
        --content_weight 1e0 \
        --image_size 512 \
        --tv_weight 1e-3
 


