#!/bin/bash 
python TumorSegmentationMaps.py \
    --rootdir /home/mathiane/LNENWork/CFlow_mathiane/CFlow/TumorSegmentation_PHH3_300323/inference/ \
    --table AnomayTable_190423.csv \
    --path_to_tiles /home/mathiane/LNENWork/PPH3_Tiles_256_256 \
    --folder_WSI_jpg /home/mathiane/LNENWork/PPH3_FullSlidesToJpg \
    --patient_id TNE2026 \
    --threshold 1.3441 \
    --tiles_size 512 \
    --outputdir ./Example_SegmentationMap_PHH3 \
    --scores MeanScoreAnomalyMap \
    --copy_discriminant_tiles

