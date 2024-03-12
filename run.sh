#for i in {0..299}; 
#do  ln -r -s /root/hdd/bbangsik/datasets/STG/data/flame_steak_colmap/colmap_$i /root/hdd/bbangsik/datasets/STG/data/flame_steak/; 
#done;
#CUDA_VISIBLE_DEVICES=3 python train.py --eval --config configs/n3d_full/flame_steak.json  --model_path /root/hdd/bbangsik/checkpoints/SpacetimeGaussians/flame_steak --source_path /root/hdd/bbangsik/datasets/STG/data/flame_steak/colmap_0
CUDA_VISIBLE_DEVICES=0 python test.py --eval --skip_train --valloader colmapvalid --config configs/n3d_full/flame_steak.json  --model_path /root/hdd/bbangsik/checkpoints/SpacetimeGaussians/flame_steak --source_path /root/hdd/bbangsik/datasets/STG/data/flame_steak/colmap_0
#rm -r  /root/hdd/bbangsik/datasets/STG/data/flame_steak/colmap*