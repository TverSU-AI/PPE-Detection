python3 demos/demo_inference.py \
  --cfg ../AlphaPose/configs/halpe_68_noface/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml \
  --checkpoint ../AlphaPose/pretrained_models/noface_fast50_dcn_combined_256x192.pth \
  --indir examples/demo/2/ --webcam 0 --vis --format coco --vis_fast --posebatch 64 \
  --detbatch 8 \
  --model_cfg ../AlphaPose/detector/yolo/cfg/yolov3-spp.cfg \
  --model_weights ../AlphaPose/detector/yolo/data/yolov3-spp.weights