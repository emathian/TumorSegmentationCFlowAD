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
- Run code by selecting class name, feature extractor, input size, flow model etc.
- The commands below should reproduce our reference MVTec results using WideResnet-50 extractor:
```
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name bottle
python3 main.py --gpu 0 --pro -inp 256 --dataset mvtec --class-name cable
python3 main.py --gpu 0 --pro -inp 256 --dataset mvtec --class-name capsule
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name carpet
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name grid
python3 main.py --gpu 0 --pro -inp 256 --dataset mvtec --class-name hazelnut
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name leather
python3 main.py --gpu 0 --pro -inp 256 --dataset mvtec --class-name metal_nut
python3 main.py --gpu 0 --pro -inp 256 --dataset mvtec --class-name pill
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name screw
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name tile
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name toothbrush
python3 main.py --gpu 0 --pro -inp 128 --dataset mvtec --class-name transistor
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name wood
python3 main.py --gpu 0 --pro -inp 512 --dataset mvtec --class-name zipper
```

## Testing Pretrained Models
- Download pretrained weights from [Google Drive](https://drive.google.com/drive/folders/1u_DupllCxl1yWvKjf_T6HMPnBoV7cV7o?usp=sharing)
- The command below should reproduce MVTec results using light-weight MobileNetV3L extractor (AUROC, AUPRO) = (98.38%, 94.72%):
```
python3 main.py --gpu 0 --pro -enc mobilenet_v3_large --dataset mvtec --action-type norm-test -inp INPUT --class-name CLASS --checkpoint PATH/FILE.PT
```

## CFLOW-AD Architecture
![CFLOW-AD](./images/fig-cflow.svg)

## Reference CFLOW-AD Results for MVTec
![CFLOW-AD](./images/fig-table.svg)
