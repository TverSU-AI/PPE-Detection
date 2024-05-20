python3 demos/demo_inference.py \
    --cfg ../AlphaPose/configs/halpe_68_noface/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml \
    --checkpoint ../AlphaPose/pretrained_models/noface_fast50_dcn_combined_256x192.pth \
    --vis \
    --posebatch 64 \
    --detbatch 8 \
    --model_cfg ../AlphaPose/detector/yolo/cfg/yolov3-spp.cfg \
    --model_weights ../AlphaPose/detector/yolo/data/yolov3-spp.weights \
    --pose_track \
    --tracker_path ../AlphaPose/trackers/weights/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth \
    --video examples/demo/arlan_three_persons.mp4 \
    --vis_fast \
    --save_video \
    --qsize 32
