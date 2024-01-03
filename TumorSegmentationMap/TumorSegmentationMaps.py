import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import os
import cv2
import pandas as pd
import argparse
import shutil
import seaborn as sns
from scipy import stats
import json
from glob import glob

__author__ = "Mathian Emilie"
__email__ = "mathiane@iarc.who.int"


ap = argparse.ArgumentParser()
ap.add_argument("--rootdir", required=False, default= './VizToyTraining/Tumor', type = str, \
    help="Folder were CFlowAD inference results were saved.")
ap.add_argument("--table", required=False, default= 'results_table.csv', type = str, \
    help="Anomaly score table file name.")
ap.add_argument("--path_to_tiles",  required=False, default= '../KI67_Tiling_256_256_40x', type = str, \
                help="Root directory where the tiles are stored. Expected organization: TilesFolder>PatientID>accept>PatientID_x_y.jpg. Here path to TilesFolder is expected.")
ap.add_argument("--folder_WSI_jpg",  required=False, default= '', type = str,\
    help="Path where WSI preview are stored.")
ap.add_argument("--patient_id",  required=True,  type = str, \
    help="Patient ID for which the segmentation will be computed. Warning patient_id must match the patient_id in the tile folder.")
ap.add_argument("--threshold",  required=False, default = 1.34408,  type = float, \
    help="Threshold to classify tiles as tumor or non-tumor. Warning: This threshold must be changed if the results are unsatisfactory.")
ap.add_argument("--tiles_size",  required=False, default = 512,  type = int, \
    help="Warning tiles size should be correct to get beautiful tumor segmentation maps!")
ap.add_argument("--outputdir",  required=False,\
    default = '../TumorSegmentation_Ki67_Baseline_2809_1004/segmentation_maps' ,  \
        type = str, help="Outputdir folder.")
ap.add_argument("--scores",  required=False,\
    default = 'MeanScoreAnomalyMap' ,  \
        type = str, help="Score used to create the segmentation map, mean anomaly score strongly recommended.")
ap.add_argument('--copy_discriminant_tiles', action='store_true', default=False,
                        help='If specified the most and less discriminant tiles will be saved in the output directory.')
args = vars(ap.parse_args())
print(args)
        
        
os.makedirs(args['outputdir'], exist_ok = True)

### Get anomaly scores table
df_anom_scores = pd.read_csv(os.path.join(args["rootdir"], args['table']), sep=',')
df_anom_scores = df_anom_scores.iloc[:,1:]  
### Extract patient ID from tiles path
df_anom_scores = df_anom_scores.assign(patient_id = [df_anom_scores.iloc[i,0].split('/')[-3] for i in range(df_anom_scores.shape[0])])

threshold_list = [args['threshold']]

# Get Q1 and Q9 anomscores quantile
## Note:  Needed for color scaling
Q1 = df_anom_scores[args['scores']].quantile(0.1)
Q9 = df_anom_scores[args['scores']].quantile(0.9)

## Get anomaly scores for patient_id
df_anom_scores_sample = df_anom_scores[df_anom_scores['patient_id'] == str(args["patient_id"])]
print(df_anom_scores_sample.head())

## Add tiles coordinates
x = []
y = []
for i in range(df_anom_scores_sample.shape[0]):
    filen = df_anom_scores_sample.iloc[i,0]
    sample_c = df_anom_scores_sample.iloc[i, -1]
    x_c = int(filen.split('/')[-1].split('_')[1])
    y_c = int(filen.split('/')[-1].split('_')[-1].split('.')[0])
    x.append(x_c)
    y.append(y_c)
    
df_anom_scores_sample = df_anom_scores_sample.assign(x=x)
df_anom_scores_sample = df_anom_scores_sample.assign(y=y)

# Tile folder name for the current patient ID 
## Needed in case of small discrepancy between args["path_to_tiles"] and tiles folder name
path_main_tiles = args["path_to_tiles"]
for f in os.listdir(path_main_tiles):
    if f.find(args['patient_id']) != -1:
        folder_name = f
        break
patientid_tiles_folder = os.path.join(path_main_tiles, folder_name)

# Organize outputdir
outputdir = args['outputdir']
for sample in set(df_anom_scores_sample['patient_id']): ## Expected to be unique
    try:
        os.makedirs(os.path.join(outputdir, sample+'_heatMap'))
    except:
        print('PatientID_folder already created in outputdir')

args['outputdir'] = os.path.join(outputdir, sample+'_heatMap')
  
# Get size of the WSI for the current sample
## Get max x & y coordinates
sample_maxX_maxY = {}
xmax = 0
ymax = 0
for folder in os.listdir(patientid_tiles_folder): # expected folder={"accept", "reject"}
    tiles_p = os.path.join(path_main_tiles, folder_name, folder)
    for tiles_l in os.listdir(tiles_p):
        xmax_c = int(tiles_l.split('_')[1])
        ymax_c  = int(tiles_l.split('_')[2].split('.')[0])
        if xmax < xmax_c:
            xmax = xmax_c
        else:
            xmax = xmax
        if ymax < ymax_c:
            ymax = ymax_c
        else:
            ymax = ymax
sample_maxX_maxY[sample] = [xmax, ymax]

# Create empty martices        
im_size = args["tiles_size"] #   384 FOR HES
scores_s = args['scores']
for k in sample_maxX_maxY.keys():
    if k in list(df_anom_scores_sample['patient_id']):
        w =  tuple(sample_maxX_maxY[k])[0] + im_size
        h = tuple(sample_maxX_maxY[k])[1] + im_size        
        seq = im_size
        W = len(list(range(1, w, seq)))
        H = len(list(range(1, h, seq)))

        mat_prob_tumor =   np.empty((W*10, H*10))#-1
        mat_prob_tumor[:] =  np.NaN

        mat_prob_binary = np.empty((W*10, H*10))
        mat_prob_binary[:] =  np.NaN
        
# Fill matrix to get tumor segmentation maps
df_test_pred_s = df_anom_scores_sample
Path2Image = [] # List of tiles path
PredTumorNomal = [] # List predicted class for each tile | Class in {"Tumor", "Normal"} | Normal must be interpret as non-tumor
for thresh in threshold_list:
    for k in sample_maxX_maxY.keys():
        if k in list(df_anom_scores_sample['patient_id']):
            for i in range(df_test_pred_s.shape[0]):
                # Get tile coord
                x_ = df_test_pred_s.iloc[i,:]['x']
                y_ = df_test_pred_s.iloc[i,:]['y']
                # Get file path
                Path2Image.append(df_test_pred_s.iloc[i,:]['file_path'])
                
                # Fill matrix with mean anomaly score of the current tile
                mat_prob_tumor[x_ // im_size * 10 :x_ // im_size *10 + 10 ,  y_ // im_size * 10 :y_ // im_size * 10 + 10 ] = df_test_pred_s.iloc[\
                                                                                                                            i,df_test_pred_s.columns.get_loc(args['scores'])]
                # Fill binary matrix > Binary segmentation map
                if df_test_pred_s.iloc[i,df_test_pred_s.columns.get_loc(args['scores'])]>thresh:
                    mat_prob_binary[x_ // im_size * 10 :x_ // im_size *10 + 10 ,  y_ // im_size * 10 :y_ // im_size * 10 + 10 ] = False
                    # Add predicted class to the result list
                    PredTumorNomal.append('Tumor')
                else : 
                    mat_prob_binary[x_ // im_size * 10 :x_ // im_size *10 + 10 ,  y_ // im_size * 10 :y_ // im_size * 10 + 10 ] = True
                    # Add predicted class to the result list
                    PredTumorNomal.append('Normal')



# Copy WSI preview in the outputdir
if args['folder_WSI_jpg'] != "": #If specified
    folder_WSI_jpg = args['folder_WSI_jpg']
    for f in  os.listdir(folder_WSI_jpg):
        if f.find(args['patient_id']) != -1:
            folder_name_full_size = f
            break
    get_full_img = os.path.join(folder_WSI_jpg, folder_name_full_size)
    print('WSI preview file path  ', get_full_img)
    try:    
        im = cv2.imread(get_full_img)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        fig=plt.figure(1,figsize=(int(xmax/1000), int(ymax/1000)))
        plt.imshow(im.astype('uint8'))
        plt.title('WSI_{}'.format(k))
        fig.savefig(os.path.join(args['outputdir'],'WSI_{}.png'.format(k)), dpi=fig.dpi)
        plt.close()
    except:
        print('WSI not available')
     
# Save tumor segmentation map 
fig=plt.figure(2,figsize=(int(ymax/1000), int(xmax/1000)))
plt.matshow(mat_prob_tumor,  cmap="coolwarm",
            interpolation='none',  fignum=2, vmin=Q1, 
            vmax =Q9)
mtitle = 'Tumor map {} '.format(args['patient_id'])
plt.title(mtitle)
plt.colorbar()
fig.savefig(os.path.join(args['outputdir'],'TumorMap_colorscaled_{}_{}.png'.format(args['patient_id'], scores_s)), dpi=fig.dpi)
plt.colorbar()
plt.close()

# Save binary tumor segmentation map 
color_map = matplotlib.colors.ListedColormap(['mediumturquoise', 'coral'])
fig = plt.figure(3, figsize=(int(ymax/1000), int(xmax/1000)))
plt.imshow(mat_prob_binary, cmap = color_map)
fig.savefig(os.path.join(args['outputdir'],
                            'Binary_segmentation_map_{}_{}_thresh_{}.png'.format(args['patient_id'], scores_s, round(thresh,3))), dpi=fig.dpi)
plt.colorbar()
plt.close()

# Save tumor segmentation map | Color not scaled
fig=plt.figure(4,figsize=(int(ymax/1000), int(xmax/1000)))
plt.matshow(mat_prob_tumor,  cmap="coolwarm",
            interpolation='none',  fignum=4)
mtitle = 'Normal tiles scores sample {} '.format(k)
plt.title(mtitle)
plt.colorbar()
fig.savefig(os.path.join(args['outputdir'],
                            'TumorMap_colorNOTscaled_{}_{}.png'.format(args['patient_id'], scores_s)), dpi=fig.dpi)
plt.colorbar()
plt.close()

# Save predicted class
df_pred = pd.DataFrame()
df_pred['file_path'] = Path2Image
df_pred['PredTumorNomal'] = PredTumorNomal
df_pred.to_csv(os.path.join(args['outputdir'], f"prediction_tumor_normal_{args['patient_id']}_{round(thresh,3)}.csv"))
    
    
# Copy most and less discriminant tiles 
if args['copy_discriminant_tiles']:
    os.makedirs(os.path.join(args['outputdir'], 'MostDiscrimiant_NonTumor'), exist_ok = True)
    os.makedirs(os.path.join(args['outputdir'],'LessDiscrimiant_Tumor'), exist_ok=True)

    df_test_pred_s = df_test_pred_s.sort_values(args['scores'])
    pos_slash = []
    pos = 0
    for c in list(df_test_pred_s['file_path'])[0]:
        if c == "/":
            pos_slash.append(pos)
        pos += 1

    for ele in list(df_test_pred_s['file_path'][:20]):
        try:
            nname  =  os.path.join(args["path_to_tiles"] , args['patient_id'] + ele[pos_slash[-2]:])
            shutil.copy(nname, os.path.join(args['outputdir'],'MostDiscrimiant_NonTumor'))
        except:
            print(f"{ele} not found")


    df_test_pred_s = df_test_pred_s.sort_values(args['scores'], ascending = False)
    for ele in list(df_test_pred_s['file_path'][:20]):
        try:
            nname  =  os.path.join(args["path_to_tiles"] , args['patient_id'] + ele[pos_slash[-2]:])
            shutil.copy(nname, os.path.join(args['outputdir'],'LessDiscrimiant_Tumor'))
        except:
            print(f"{ele} not found")