#!/usr/bin/env python
# coding: utf-8
###################
# by: Mojtaba Jafaritadi, PhD
###################

import numpy as np
import pandas as pd
from model_args import get_args
import datetime
from models import Generator3D
from inference_with_Bootstrapping import denoise_Gated_PET_Bootstrap,denoise_NonGated_PET_Bootstrap

print("\n * Python libraries loaded successfully... * \n")

print("\n * START ENVIRONMENT CONFIGURATION* \n")
import os
#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
os.chdir('/home/mojjaf/Pix2Pix_CardiacPET')

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[3:4], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
print("\n * Environment and GPU Setup compelted... * \n")


# In[2]:
print("Available devices:")
for i, device in enumerate(tf.config.list_logical_devices()):
  print("%d) %s" % (i, device))

dataset= "cardiac_PET"
mode='3D'  #2.5 D or single (2D)
experiment = "/experiments/Bootstrap_Pix2Pix_"+dataset+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"  # 
model_name = "Bootstrap_trainedmodel_3D_"+dataset+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/" 
print(f"\nExperiment: {experiment}\n")

args = get_args()
project_dir = args.main_dir

experiment_dir = project_dir+experiment+mode

if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir) 
    
output_preds_3D = experiment_dir+'Pix2Pix_saved_predictions/'

if not os.path.exists(output_preds_3D):
    os.makedirs(output_preds_3D)

pair_list=pd.read_csv('/home/mojjaf/Pix2Pix_CardiacPET/data/pairs_low2high_subsetlist.csv',dtype=str,delimiter=';') 
train_list=pair_list['Train_NG'].dropna()
val_list=pair_list['Validation_NG'].dropna()
test_list=pair_list['Test_NG'].dropna()
train_subjects=train_list.tolist()
val_subjects=val_list.tolist()
test_subjects=test_list.tolist()

pair_list
print("\n * Patient list loaded and data subsets created... * \n")
pair_list


data_dir=args.data_dir
checkpoint_dir = args.model_path



print("\n * Starting inference for Non-Gated mode... * \n")

model_dir=checkpoint_dir+'/Generator3D.h5'
model_generator=Generator3D()
model_dim='3D'
pet_dose_list=['05','10','15']
image_size=128
experiment_dir_gated_bl=experiment_dir+'/gated_csvs/baseline/'
if not os.path.exists(experiment_dir_gated_bl):
    os.makedirs(experiment_dir_gated_bl)

experiment_dir_gated_pr=experiment_dir+'/gated_csvs/predictions/'
if not os.path.exists(experiment_dir_gated_pr):
    os.makedirs(experiment_dir_gated_pr)

experiment_dir_nongated_bl=experiment_dir+'/nongated_csvs/baseline/'
if not os.path.exists(experiment_dir_nongated_bl):
    os.makedirs(experiment_dir_nongated_bl)

experiment_dir_nongated_pr=experiment_dir+'/nongated_csvs/predictions/'
if not os.path.exists(experiment_dir_nongated_pr):
    os.makedirs(experiment_dir_nongated_pr)

fig_plt=False
for dose in pet_dose_list:
    print(dose)
    analysis_report_nongated_baseline,analysis_report_nongated_preds =denoise_NonGated_PET_Bootstrap(data_dir,'all_NG2_low', 'all_NG2',model_dir,model_generator,model_dim,dose,image_size, test_subjects,output_preds_3D,fig_plt)
    
    analysis_report_nongated_baseline.to_csv(experiment_dir_nongated_bl+'_baselineresults_nongated_'+dose+'perc'+'.csv', index=False)
    analysis_report_nongated_preds.to_csv(experiment_dir_nongated_pr+'_predictionresults_nongated_'+dose+'perc'+'.csv', index=False)

    print("\n * Bootstrap summary (predicted Non-Gated PET): * \n")

    snr_cum=np.array(analysis_report_nongated_preds['psnr']).astype(float)### predicted vs high dose PET
    ssi_cum=np.array(analysis_report_nongated_preds['ssim']).astype(float)
    nrmse_cum=np.array(analysis_report_nongated_preds['nrmse']).astype(float)

    ########## Mean values of the PSNR, SSIM, and MSE #########
    snr_resPR_mean = np.mean(snr_cum)
    ssim_resPR_mean = np.mean(ssi_cum)
    mse_resPR_mean = np.mean(nrmse_cum)

    ########## Standard deviation values of the PSNR, SSIM, MSE#########
    snr_resPR_std = np.std(snr_cum)
    ssim_resPR_std = np.std(ssi_cum)
    mse_resPR_std = np.std(nrmse_cum)

    print("Mean of the PSNR is ", snr_resPR_mean, "+/-", snr_resPR_std) 
    print("Mean of the SSIm is ", ssim_resPR_mean,"+/-",ssim_resPR_std) 
    print("Mean of the NRMSE is",mse_resPR_mean,"+/-",mse_resPR_std)


print("\n * Bootstrap mode (non-Gated) successfully COMPLETED. Saving the results... * \n")

#analysis_report

print("\n * Starting bootstrap for Dual Gated mode... * \n")

all_gates=['_0821','_0822','_0823','_0824','_0825','_0826','_0827','_0828','_0829',
       '_0830','_0831','_0832','_0833','_0834','_0835','_0836','_0836','_0836',
       '_0837','_0838','_0839','_0840','_0841','_0842','_0843','_0844','_0845']


model_dir=checkpoint_dir+'/Generator3D.h5'
model_generator=Generator3D()
model_dim='3D'
image_size=128
for gate in all_gates:
    #gate='_0845'##bin 16
    fig_plt=False
    analysis_report_gated_baseline,analysis_report_gated_preds =denoise_Gated_PET_Bootstrap(data_dir,'gated', 'all_NG2',gate,model_dir,model_generator,model_dim,image_size,test_subjects,output_preds_3D,fig_plt)

    analysis_report_gated_baseline.to_csv(experiment_dir_gated_bl+'_baselineresults'+gate+'_'+'.csv', index=False)
    analysis_report_gated_preds.to_csv(experiment_dir_gated_pr+'_predictionresults'+gate+'_'+'.csv', index=False)



    snr_cum=np.array(analysis_report_gated_preds['psnr']).astype(float)### predicted vs high dose PET

    ssi_cum=np.array(analysis_report_gated_preds['ssim']).astype(float)
    nrmse_cum=np.array(analysis_report_gated_preds['nrmse']).astype(float)


    ########## Mean values of the PSNR, SSIM, and MSE #########
    snr_resPR_mean = np.mean(snr_cum)
    ssim_resPR_mean = np.mean(ssi_cum)
    mse_resPR_mean = np.mean(nrmse_cum)

    ########## Standard deviation values of the PSNR, SSIM, MSE#########
    snr_resPR_std = np.std(snr_cum)
    ssim_resPR_std = np.std(ssi_cum)
    mse_resPR_std = np.std(nrmse_cum)

    print("\n * Inference summary (predicted Gated PET): ",gate,"* \n")
    print("Mean of the PSNR is ", snr_resPR_mean, "+/-", snr_resPR_std) 
    print("Mean of the SSIM is ", ssim_resPR_mean,"+/-",ssim_resPR_std) 
    print("Mean of the NRMSE is",mse_resPR_mean,"+/-",mse_resPR_std)

print("\n * Bootstrap mode successfully COMPLETED. Saved all gated results. * \n")


# In[ ]:




