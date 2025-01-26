from __future__ import print_function
import os, time, random, math
import numpy as np
import torch
import datetime
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from skimage.measure import label, regionprops
from tqdm import tqdm
from config import get_args
from visualize import *
from model import load_decoder_arch, load_encoder_arch, positionalencoding2d, activation
from utils import *
from custom_datasets import *
from custom_models import *
import pandas as pd
from PIL import Image
## parallel
import hostlist
import torch.distributed as dist
from ignite.contrib import metrics
from torch.nn.parallel import DistributedDataParallel as DDP

gamma = 0.0
theta = torch.nn.Sigmoid()
log_theta = torch.nn.LogSigmoid()

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def train_meta_epoch(c, epoch, loader, encoder, decoders, optimizer, pool_layers, N):
    P = c.condition_vec
    L = c.pool_layers
    decoders = [decoder.train() for decoder in decoders]
    adjust_learning_rate(c, optimizer, epoch)
    I = len(loader)
    iterator = iter(loader)
    for sub_epoch in range(c.sub_epochs):
        print('Epoch: {:d} \t sub-epoch: {:.4f} '.format(epoch, sub_epoch))
        train_loss = 0.0
        train_count = 0
        
        for i in range(I):
            if i % 100 == 0:
                print('step  % : ', (i/I) * 100, ' i/I = ', i , '/' , I)               
            # warm-up learning rate
            lr = warmup_learning_rate(c, epoch, i+sub_epoch*I, I*c.sub_epochs, optimizer)
            # sample batch
            try:
                image, _, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                image, _, _, _ = next(iterator)
            # encoder prediction
            image = image.to(c.device)  # single scale
            with torch.no_grad():
                _ = encoder(image)
            # train decoder
            e_list = list()
            c_list = list()
            for l, layer in enumerate(pool_layers):
                e = activation[layer].detach()  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S    
                #
                p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                perm = torch.randperm(E).to(c.device)  # BHW
                decoder = decoders[l]
                #
                FIB = E//N  # number of fiber batches
                assert FIB > 0, 'MAKE SURE WE HAVE ENOUGH FIBERS, otherwise decrease N or batch-size!'
                for f in range(FIB):  # per-fiber processing
                    idx = torch.arange(f*N, (f+1)*N)
                    c_p = c_r[perm[idx]]  # NxP
                    e_p = e_r[perm[idx]]  # NxC
                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p,])
                    else:
                        z, log_jac_det = decoder(e_p)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_theta(log_prob)
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()
                    train_loss += t2np(loss.sum())
                    train_count += len(loss)

            # Save results
            ## Parallel
            if c.parallel:
                epoch_s = str(epoch)
                sub_epoch_s = str(sub_epoch)
                os.makedirs(c.weights_dir, exist_ok = True )
                os.makedirs(os.path.join(c.weights_dir, c.class_name), exist_ok = True)
                os.makedirs(os.path.join(c.weights_dir, c.class_name, epoch_s), exist_ok = True)

                for j, ddp_decoder in enumerate(decoders):
                    if i % 5000 == 0:
                        mean_train_loss = train_loss / train_count
                        print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_train_loss, lr))
                        filename = '{}_mataepoch_{}_subepoch_{}_loader_{}_decoder_{}.pt'.format(c.model, epoch_s, sub_epoch_s, i,j)
                        path = os.path.join(c.weights_dir, c.class_name, epoch_s,  filename)
                        print('Path : ', path)
                        if c.parallel:
                            if c.idr_torch_rank == 0:
                                torch.save(ddp_decoder.state_dict(), path)
                        else:
                            torch.save(ddp_decoder.state_dict(), path)
    
        mean_train_loss = train_loss / train_count
        if c.parallel:
            if c.verbose:
                if c.idr_torch_rank == 0:
                    print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_train_loss, lr))
            ## TO IMPLEMENT SAVE PARALLEL
        else:
            if c.verbose:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_train_loss, lr))
            save_weights_epoch(c, encoder, decoders, c.model, epoch, sub_epoch) 


def write_anom_map(c, super_mask, files_path_list_c, original_images, threshold=0.5):
    """
    Write segmented anomaly maps to disk with red and blue colors.

    Parameters:
    - c: Configuration object containing settings.
    - super_mask: The super-pixel mask of the anomalies (numpy array or tensor).
    - files_path_list_c: List of file paths for saving the maps.
    - original_images: List of original images (PIL Images or numpy arrays).
    - threshold: Threshold value for binary segmentation (default=0.5).
    """

    # Ensure the output directory exists
    output_dir = c.viz_dir  # Assuming this is defined in your configuration
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, file_path in enumerate(files_path_list_c):
        # Convert super_mask to numpy array if it's a tensor
        img_np = super_mask if isinstance(super_mask, np.ndarray) else super_mask.cpu().numpy()

        # If super_mask is 3D (batch of images), select the corresponding image
        if img_np.ndim == 3:
            img_np = img_np[idx]

        # Apply threshold to create binary segmentation
        binary_mask = (img_np > threshold).astype(np.uint8)

        # Get the original image
        original_img = original_images[idx]
        if isinstance(original_img, Image.Image):
            original_img = np.array(original_img)  # Convert PIL Image to numpy array

        # Create an RGB version of the original image
        if original_img.ndim == 2:  # Grayscale image
            original_img = np.stack([original_img] * 3, axis=-1)
        elif original_img.shape[2] == 1:  # Single-channel image
            original_img = np.concatenate([original_img] * 3, axis=-1)

        # Create a colored overlay for the anomaly map
        overlay = np.zeros_like(original_img)

        # Color anomalies in red (255, 0, 0)
        overlay[binary_mask == 1] = [255, 0, 0]

        # Color normal regions in blue (0, 0, 255)
        overlay[binary_mask == 0] = [0, 0, 255]

        # Blend the original image with the overlay
        alpha = 0.5  # Transparency factor for the overlay
        segmented_img = (original_img * (1 - alpha) + overlay * alpha).astype(np.uint8)

        # Convert to PIL Image and save
        segmented_img_pil = Image.fromarray(segmented_img)
        output_path = os.path.join(output_dir, f"{os.path.basename(file_path)}_segmented.png")
        segmented_img_pil.save(output_path)

        print(f"Saved segmented image: {output_path}")
    
    
def test_meta_epoch_lnen(c, epoch, loader, encoder, decoders, pool_layers, N):
    # test
    print('\nCompute loss and scores on test set:')
    #
    P = c.condition_vec
    decoders = [decoder.eval() for decoder in decoders]
    height = list()
    width = list()
    test_loss = 0.0
    test_count = 0
    start = time.time()
    score_label_mean_l = []
    I = len(loader)
    os.makedirs(os.path.join(c.viz_dir, c.class_name), exist_ok= True)
    if not c.infer_train:
        res_tab_name = 'results_table.csv'
    else:
        res_tab_name = 'results_table_train.csv'
    print('os.path.join(c.viz_dir, c.class_name, res_tab_name)  ', os.path.join(c.viz_dir, c.class_name, res_tab_name))
    with open(os.path.join(c.viz_dir, c.class_name, res_tab_name), 'w') as table_file: 
        table_file.write("file_path,binary_lab,MaxScoreAnomalyMap,MeanScoreAnomalyMap\n")
    table_file.close()
    with torch.no_grad():
        for i, (image, label, mask, filespath) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            if i % 1000 == 0:
                print('\n test_meta_epoch_lnen - step  % : ', (i/I) * 100, ' i/I = ', i , '/' , I)
            files_path_list_c = filespath
            # save
            
            labels_c = t2np(label)
            # data
            
            image = image.to(c.device) # single scale
            _ = encoder(image)  # BxCxHxW
            # test decoder
            e_list = list()
            test_dist = [list() for layer in pool_layers]
            test_map = [list() for p in pool_layers]
            for l, layer in enumerate(pool_layers):
                e = activation[layer]  # BxCxHxW

                B, C, H, W = e.size()
                S = H*W
                E = B*S
                #
                if i == 0:  # get stats
                    height.append(H)
                    width.append(W)
                #
                p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                #
                m = F.interpolate(mask, size=(H, W), mode='nearest')
                m_r = m.reshape(B, 1, S).transpose(1, 2).reshape(E, 1)  # BHWx1
                #
                decoder = decoders[l]
                FIB = E//N + int(E%N > 0)  # number of fiber batches
                for f in range(FIB):
                    if f < (FIB-1):
                        idx = torch.arange(f*N, (f+1)*N)
                    else:
                        idx = torch.arange(f*N, E)
                    #
                    c_p = c_r[idx]  # NxP
                    e_p = e_r[idx]  # NxC
                    m_p = m_r[idx] > 0.5  # Nx1
                    #
                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p,])
                    else:
                        z, log_jac_det = decoder(e_p)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_theta(log_prob)
                    test_loss += t2np(loss.sum())
                    test_count += len(loss)
                    test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()

            test_map = [list() for p in pool_layers]
            for l, p in enumerate(pool_layers):
                test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1
                test_norm-= torch.max(test_norm) # normalize likelihoods to (-Inf:0] by subtracting a constant
                test_prob = torch.exp(test_norm) # convert to probs in range [0:1]
                test_mask = test_prob.reshape(-1, height[l], width[l])
                test_mask = test_prob.reshape(-1, height[l], width[l])

                # upsample
                test_map[l] = F.interpolate(test_mask.unsqueeze(1),
                    size=c.crp_size, mode='bilinear', align_corners=True).squeeze().numpy()
            # score aggregation
            score_map = np.zeros_like(test_map[0])
            for l, p in enumerate(pool_layers):
                score_map += test_map[l]
            score_mask = score_map
            super_mask = score_mask

            if super_mask.ndim == 2:
                score_label_max = np.max(super_mask, axis=(0, 1))
                score_label_mean = np.mean(super_mask, axis=(0, 1))
            elif super_mask.ndim == 3:
                score_label_max = np.max(super_mask, axis=(1, 2))
                score_label_mean = np.mean(super_mask, axis=(1, 2))
            else:
                raise ValueError(f"Unexpected dimensions for super_mask: {super_mask.shape}")
            
            ### write table 
            res_df = pd.DataFrame()
            res_df['FilesPath'] = files_path_list_c
            res_df['BinaryLabels'] = labels_c
            res_df['MaxScoreAnomalyMap'] = score_label_max.flatten().tolist()
            res_df['MeanScoreAnomalyMap'] = score_label_mean.flatten().tolist()
            with open(os.path.join(c.viz_dir, c.class_name, res_tab_name), 'a') as table_file: 
                for row in range(res_df.shape[0]):
                    file_path_ = res_df[ 'FilesPath'][row]
                    binary_lab_ = res_df['BinaryLabels'][row]
                    MaxScoreAnomalyMap = res_df[ 'MaxScoreAnomalyMap'][row]
                    MeanScoreAnomalyMap = res_df[ 'MeanScoreAnomalyMap'][row]
                    table_file.write(f"{file_path_},{binary_lab_},{MaxScoreAnomalyMap},{MeanScoreAnomalyMap}\n")
                table_file.close()
            if c.viz_anom_map:
                # Convert images to numpy arrays if they are tensors
                original_images = [t2np(img) for img in image]  # Convert batch of images to numpy

                write_anom_map(c, super_mask, files_path_list_c, original_images, threshold=0.5)
            if i % 1000 == 0 :
                print('Epoch: {:d} \t step: {:.4f} '.format(epoch, i))
        

def main(c):
    ## Extract config ###############################################
    # model definition
    c.model = "{}_{}_{}_pl{}_cb{}_inp{}_run{}_{}".format(
            c.dataset, c.enc_arch, c.dec_arch, c.pool_layers, c.coupling_blocks, c.input_size, c.run_name, c.class_name)
    # image format
    c.img_size = (c.input_size, c.input_size)  # HxW format
    c.crp_size = (c.input_size, c.input_size)  # HxW format
    c.norm_mean, c.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    c.img_dims = [3] + list(c.img_size)
    # network hyperparameters
    c.clamp_alpha = 1.9  # see paper equation 2 for explanation
    c.condition_vec = 128
    c.dropout = 0.0  # dropout in s-t-networks
    # dataloader parameters
    if  c.dataset == 'TumorNormal':
        print(f"c.dataset = {c.dataset}")
    # To extend for other dataset
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))
    # output settings
    c.verbose = True
    c.hide_tqdm_bar = True
    c.save_results = True
    # unsup-train
    c.print_freq = 2
    c.temp = 0.5
    # Learning rate config
    c.lr_decay_epochs = [i*c.meta_epochs//100 for i in [50,75,90]]
    print('LR schedule: {}'.format(c.lr_decay_epochs))
    c.lr_decay_rate = 0.1
    c.lr_warm_epochs = 2
    c.lr_warm = True
    c.lr_cosine = True
    if c.lr_warm:
        c.lr_warmup_from = c.lr/10.0
        if c.lr_cosine:
            eta_min = c.lr * (c.lr_decay_rate ** 3)
            c.lr_warmup_to = eta_min + (c.lr - eta_min) * (
                    1 + math.cos(math.pi * c.lr_warm_epochs / c.meta_epochs)) / 2
        else:
            c.lr_warmup_to = c.lr
    # Init GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = c.gpu
    c.use_cuda = not c.no_cuda and torch.cuda.is_available()
    init_seeds(seed=int(time.time()))
    c.device = torch.device("cuda" if c.use_cuda else "cpu")
   
    ###########################################################################
    # Parallel training
    # Warning: only available for training!
    # To speed up inference, divide your test set into subsets
    # and run the process once on each cut. 
    if c.parallel :
        # GPU distribution
        idr_torch_rank = int(os.environ['SLURM_PROCID'])
        # New config
        c.idr_torch_rank = idr_torch_rank
        local_rank = int(os.environ['SLURM_LOCALID'])
        idr_torch_size = int(os.environ['SLURM_NTASKS'])
        cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
        torch.backends.cudnn.enabled = False
        # get node list from slurm
        hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
        gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")
        # define MASTER_ADD & MASTER_PORT
        os.environ['MASTER_ADDR'] = hostnames[0]
        os.environ['MASTER_PORT'] = str(12456 + int(min(gpu_ids))); #Avoid port conflits in the node #str(12345 + gpu_ids)
        dist.init_process_group(backend='nccl', 
                            init_method='env://', 
                            world_size=idr_torch_size, 
                            rank=idr_torch_rank)
        torch.cuda.set_device(local_rank)
    # Define device
    gpu = torch.device("cuda")
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    
    # DL network configuration
    L = c.pool_layers # number of pooled layers
    encoder, pool_layers, pool_dims = load_encoder_arch(c, L)
    encoder = encoder.to(gpu).eval()
    if  c.parallel: ## Load on GPUs
        ddp_encoder = DDP(encoder, device_ids=[local_rank]) # , output_device=local_rank
    # NF decoder
    decoders = [load_decoder_arch(c, pool_dim) for pool_dim in pool_dims]
    decoders = [decoder.to(gpu) for decoder in decoders]
    if c.parallel:  ## Load on GPUs
        ddp_decoders = []
        for decoder in decoders:
             ddp_decoders.append(DDP(decoder, device_ids=[local_rank])) # , output_device=local_rank
    
    params = list(decoders[0].parameters())
    for l in range(1, L):
        if c.parallel:
            params += list(ddp_decoders[l].parameters())
        else:
            params += list(decoders[l].parameters())

    # optimizer
    optimizer = torch.optim.Adam(params, lr=c.lr)
    # Workers
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}

    ## Create data loader ###############################################
    if c.action_type == 'norm-train':
        train_dataset = TumorNormalDataset(c, is_train=True)
    else: # Inference - Warning Parallel inference not implemented 
          test_dataset  = TumorNormalDataset(c, is_train=False)
    # Parallel data loader
    if c.parallel and c.action_type == 'norm-train': 
        batch_size_per_gpu =  c.batch_size // idr_torch_size
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=idr_torch_size, rank=idr_torch_rank) 
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,  batch_size=batch_size_per_gpu,  shuffle=False,   num_workers=0,  pin_memory=True, sampler=train_sampler)
    # Single GPU data loader
    else:
        if c.action_type == 'norm-train':
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, drop_last=True, **kwargs)
        else: # Inference
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=True, drop_last=False, **kwargs)

    N = 256  # hyperparameter that increases batch size for the decoder model by N
    
    # Inference/Training loop 
    if c.action_type == 'norm-test':
        c.meta_epochs = 1
    for epoch in range(c.meta_epochs): 
        ## Inference 
        if c.action_type == 'norm-test' and c.checkpoint:
            if c.parallel:
                ## Parallel inference
                print("Load weights Parallel")
                ## Load the n decoder
                for i, ddp_decoder in enumerate(ddp_decoders):
                    c_checkpoint = c.checkpoint[:-3]+f'_{i}.pt'
                    print(c_checkpoint)
                    ddp_decoder.load_state_dict(torch.load(c_checkpoint))
                    print('EVAL IN C.PARALLEL test_meta_epoch_lnen')
                    ## Run inference
                    test_meta_epoch_lnen(c, epoch, test_loader, ddp_encoder, ddp_decoders, pool_layers, N) 
            else:
                ## Not parallel inference
                load_weights(encoder, decoders, c.checkpoint)
                test_meta_epoch_lnen(c, epoch, test_loader, encoder, decoders, pool_layers, N) # test_meta_epoch_lnen

        ## Training    
        elif c.action_type == 'norm-train':
            if c.parallel:
                train_meta_epoch(c, epoch, train_loader, ddp_encoder, ddp_decoders, optimizer, pool_layers, N)
            else:
                train_meta_epoch(c, epoch, train_loader, encoder, decoders, optimizer, pool_layers, N)
        else:
            raise NotImplementedError('{} is not supported action type!'.format(c.action_type))               

if __name__ == '__main__':
    c = get_args()
    main(c)
