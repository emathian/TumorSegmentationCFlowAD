![PresentationImg](ImgPresentation2.png)
## Tumor segmentation with CFLOW-AD: 
Unsupervised tumor segmentation method using an adapted version of CFlow, an anomaly detection model.
This model was trained to learn the distribution of tumor tiles extracted from whole slide images (WSI), so that non-tumor areas could be detected at the time of inference.

- Original article: D. Gudovskiy [CFlow-AD](https://openaccess.thecvf.com/content/WACV2022/papers/Gudovskiy_CFLOW-AD_Real-Time_Unsupervised_Anomaly_Detection_With_Localization_via_Conditional_Normalizing_WACV_2022_paper.pdf), WACV 2022.
- Original code: [https://github.com/gudovskiy/cflow-ad](https://github.com/gudovskiy/cflow-ad)
- Aplication inspired from E. Mathian, [HaloAE](https://www.scitepress.org/PublishedPapers/2023/118659/118659.pdf) Visapp 2023 
- Method used for tumor segmentation tasks in "Assessment of the current and emerging criteria for the histopathological classification of lung neuroendocrine tumours in the lungNENomics project." ESMO Open 2023 (under review)

## Installation
- Clone this repository: tested on Python 3.8
- Install [PyTorch](http://pytorch.org/): tested on v2.1.2
- Install [Torchvison](https://pytorch.org/vision/stable/index.html) tested on v0.16.2
- Install [Timm](https://timm.fast.ai/) tested on v0.6.11
- Install [cudatoolkit](https://developer.nvidia.com/cuda-toolkit) tested on 11.8.0
- Install [pytorch-cuda](https://pytorch.org/get-started/locally/) tested on 11.8
- Install [scikit-image](https://scikit-image.org/) tested on 0.19.3
- Install [scikit-learn](https://scikit-learn.org/stable/) tested on 1.3.0
- Install [pillow](https://pillow.readthedocs.io/en/stable/)  tested on 10.0.1
- Install any version of pandas, numpy, matplotlib
- For simplicity [FrEIA Flows](https://github.com/VLL-HD/FrEIA): tested on [the recent branch](https://github.com/VLL-HD/FrEIA/tree/4e0c6ab42b26ec6e41b1ee2abb1a8b6562752b00) has already be cloned in this repository
- Other dependencies in environment.yml

Install all packages with this command:
```
$ conda env create -f environment.yml
```

## Datasets
This method has been tested for 3 types of histological images:
+ Haematoxylin and Eosin (HE) | Haematoxylin, Eosin Saffron (HES) stained WSI:
    + Number of tumor tiles (for train and test) = 12,991 (69 patients)
    + Number of non-tumor tiles (for test) = 4,815 (33 patients)
+ Ki-67 immunohistochemical stained WSI:
    + Number of tumor tiles (for train and test) = 19,053 (77 patients)
    + Number of non-tumor tiles (for test) = 10,259 (40 patients)
+ Phosphohistone histone H3 (PHH3)-stained WSIs can be segmented using Ki-67 tumor tiles as a training set.

**These two dataset are available on request from mathiane[at]iarc[dot]who[dot]int and will soon be available online.**

## Code Organization
- ./custom_datasets - contains dataloaders for TumorNormalDataset :
    - The dataloader is based on a file listing the path to the tiles.
    -  Examples: `./Datasets/ToyTrainingSetKi67Tumor.txt` and `./Datasets/ToyTestSetKi67Tumor.txt`

- ./custom_models 
    - contains pretrained `resnet` feature extractors:
        - For the tumor segmentations tasks we used a wide-Resnet 50 (see: `resnet.py` line 352)
        -  *Note: additional features extrators can be found in the original [CFlow AD repository](https://github.com/gudovskiy/cflow-ad)*
    - the `utils` contains functions to save and load the checkpoint


- ./FrEIA - clone from [https://github.com/VLL-HD/FrEIA](https://github.com/VLL-HD/FrEIA) repository.

- models - Build encoder and decoder
    - The encoder is based on a pretrained resnet (see: `custom_models/resnet.py`)
    - The decoder is based on FrEIA modules

- main: Main script to train and test the model.

## Training Models
- An example of the configurations used to segment HE/HES, Ki-67 and PHH3 WSI is available in `Run/Train/TumorNormal/TrainToyDataKi67.sh`
- *Configs can be viwed in `config.py`*
- The commands below are used to train the model based on the toy data set:
```
bash Run/Train/TumorNormal/TrainToyDataKi67.sh
```
- **Warnings: Network weights will be saved for all epochs in `config.weights-dir/config.class-name/meta-epoch/ModelName_ClassName_MetaEpoch_SubEpoch.pt`. Each checkpoint creates is associated 903MB file.**

## Testing Pretrained Models
- Download pretrained weights are available on request and will be soon available online 
- An example of the configurations used to infer the test set is gien in `Run/Test/TumorNormal/TestToyDataset.sh`
```
bash Run/Test/TumorNormal/TestToyDataset.sh
```
- Main configurations:
    + checkpoint: Path to model weights to be loaded to infer the test tiles.
    + viz-dir: Directory where the result table will be saved.
    + viz-anom-map: If specified, all anomaly maps will be written to the `viz-dir` directory in `.npy` format.

## Results exploration
For each tile, `results_table.csv` summarises:
- Its path, which may include the patient ID
- Binary tile labels, useful for sorted datasets: Tumour = 2 and Non-tumour = 1 
- Max anomaly scores: value of the highest anomaly score of the tile
- Mean anomaly scores: average anomaly score of the tile

**The distributions of these score are used to segment the WSI.**

An example of result exploration for the segmentation of HE/HES WSI is given in `ExploreResultsHETumorSeg.html`.

## Get tumor segmentation map 

The `TumorSegmentationMaps.py` script is used to create the tumour segmentation map for a WSI. An example configuration is given in `ExRunTumorSegmentationMap.sh`. The results of this script are stored in the `Example_SegmentationMap_PHH3` folder, which also gives an example of the model's performance in segmenting a PHH3-immunostained WSI.

## TO DO LIST

+ :construction: Check parallel training 
+ :construction: Check parallel test
+ :construction: Model checkpoints Ki-67 and HES/HE

/opt/conda/envs/TumorSegmentationWithCFlow/bin/python /build/TumorSegmentationCFlowAD/main.py --action-type norm-test --checkpoint /app/weights/TumorNormal_wide_resnet50_2_freia-cflow_pl3_cb8_inp384_run0_Tumor_10_4.pt --gpu 0 -inp 384 --dataset TumorNormal --class-name Data --list-file-test /app/static/66e9a3a0634e9e3a51f2b690/tmp/tmpmoa0xf4t/tmpdqdcfsfq.txt --viz-dir /app/static/66e9a3a0634e9e3a51f2b690/tmp/tmpmoa0xf4t --viz-anom-map
