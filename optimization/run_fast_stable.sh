
python fast_stable_texture_transfer.py \
	--content_image ./data/ava.png \
	--style_image ./data/mona.png \
	--init_image image \
	--style_scale 1 \
	--semantic 0 \
	--style_weight 1e6 \
	--content_weight 1e0 \
	--image_size 512 \
	--output_image ./output/ava-2-mona.png \
	--original_colors 0 \
        --max_nums 1000

#style_weight 1e6 for elephant-turtle
#content_weight 8e0 for elephant-turtle

