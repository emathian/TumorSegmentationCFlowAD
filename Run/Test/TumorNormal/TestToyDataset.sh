#!/bin/bash
python ../../../main.py \
		--action-type norm-test \
        --checkpoint ../../../weights/Tumor/3/TumorNormal_wide_resnet50_2_freia-cflow_pl3_cb8_inp384_run0_Tumor_3_6.pt \
		--gpu 0 \
		-inp 384 \
		--dataset TumorNormal \
		--class-name Tumor \
		--list-file-train ../../../Datasets/ToyTrainingSetKi67Tumor.txt \
		--list-file-test ../../../Datasets/ToyTestSetKi67Tumor.txt \
        --viz-dir ../../../VizToyTraining \
        --viz-anom-map


 
