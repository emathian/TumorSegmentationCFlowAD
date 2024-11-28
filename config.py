from __future__ import print_function
import argparse

__all__ = ['get_args']


def get_args():
    parser = argparse.ArgumentParser(description='Tumor segmentation CFLOW-AD')
    
    # Definition of the data set
    parser.add_argument('--dataset', default='TumorNormal', type=str, metavar='D',
                        help='dataset name: TumorNormal ')
    parser.add_argument('-cl', '--class-name', default='Tumor', type=str, metavar='C',
                        help='class name for LNEN  (default: none)')
    parser.add_argument('-lfr', '--list-file-train', default='../../../Datasets/ToyTrainingSetKi67Tumor.txt', type=str, metavar='C',
                        help='List of files for LNEN dataset')
    parser.add_argument('-lft', '--list-file-test', default='../../../Datasets/ToyTestSetKi67Tumor.txt', type=str, metavar='C',
                        help='List of files for LNEN dataset')
    parser.add_argument('-inp', '--input-size', default=384, type=int, metavar='C',
                        help='image resize dimensions (default: 256)')
    
    # Model definition
    parser.add_argument('-enc', '--enc-arch', default='wide_resnet50_2', type=str, metavar='A',
                        help='feature extractor: wide_resnet50_2')
    parser.add_argument('-wb', '--backbone_weights', default='', type=str, metavar='C', 
                        help='path to Wide resnet 50 weights if not specified \
                        https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth will be downloaded')
    parser.add_argument('-dec', '--dec-arch', default='freia-cflow', type=str, metavar='A',
                        help='normalizing flow model (default: freia-cflow)')
    parser.add_argument('-pl', '--pool-layers', default=3, type=int, metavar='L',
                        help='number of layers used in NF model (default: 3)')
    parser.add_argument('-cb', '--coupling-blocks', default=8, type=int, metavar='L',
                        help='number of layers used in NF model (default: 8)')
    
    # Training hyerparameters
    parser.add_argument('-bs', '--batch-size', default=32, type=int, metavar='B',
                        help='train batch size (default: 32)') # 64 if parallel 2 GPU
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 2e-4)') # low_lr = 2e-5
    parser.add_argument('--meta-epochs', type=int, default=25, metavar='N',
                        help='number of meta epochs to train (default: 25)')
    parser.add_argument('--sub-epochs', type=int, default=8, metavar='N',
                        help='number of sub epochs to train (default: 8)')
    parser.add_argument('--pro', action='store_true', default=False,
                        help='enables estimation of AUPRO metric')
    
    # Action mode 
    parser.add_argument("--action-type", default='norm-train', type=str, metavar='T',
                        help='norm-train/norm-test (default: norm-train) if norm test inference of test set')
    parser.add_argument('--infer-train', action='store_true', default=False, 
                        help='If specified Infer the train set')
    
    # Saving 
    ## saving checkpoints while training
    parser.add_argument('-wd', '--weights-dir', default='../../../weights/carpet_parallel_hatim_10102022', type=str, metavar='C', help='result outputdir')

    parser.add_argument('--checkpoint', default='', type=str, metavar='D',
                        help='file with saved checkpoint')
    parser.add_argument('-vd', '--viz-dir', default='/gpfsscratch/rech/uli/ueu39kt/CFLOW/viz', type=str, metavar='C',
                        help='visualization outputdir')
    parser.add_argument('-rd', '--res-dir', default='/gpfsscratch/rech/uli/ueu39kt/CFLOW/results', type=str, metavar='C',
                        help='result outputdir')
    parser.add_argument('-run', '--run-name', default=0, type=int, metavar='C',
                        help='name of the run (default: 0)')
    
    # Device setting
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='If specified run in parallel on several GPU')
    parser.add_argument('--workers', default=4, type=int, metavar='G',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--gpu", default='0', type=str, metavar='G',
                        help='GPU device number')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
   
    parser.add_argument('--viz-anom-map', action='store_true', default=False,
                       help='saves anomaly maps')
    
    #parser.add_argument('--video-path', default='.', type=str, metavar='D',
    #                    help='video file path')

    args = parser.parse_args()
    
    return args
