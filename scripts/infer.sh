python3 demos/demo_inference.py \
    --cfg ../AlphaPose/configs/halpe_68_noface/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml \
    --checkpoint ../AlphaPose/pretrained_models/noface_fast50_dcn_combined_256x192.pth \
    --image /home/throder/Загрузки/putin-t-pose.jpg \
    --save_img \
    --format coco \
    --vis_fast \
    --posebatch 64 \
    --detbatch 8
