import math
from pathlib import Path
from math import log10, sqrt
from matplotlib import pyplot as plt
import os
from IPython import display
import nibabel as nib
from skimage.transform import resize
import numpy as np
import argparse
import pandas as pd
from main_arguments import get_args
from tensorflow import keras
from IPython import display
import tensorflow as tf


def nifty_imageloader(filepath, exclude=False):
    image_3D=np.array(nib.load(filepath).dataobj)
    if exclude:
        image_3D=image_3D[exclude:-exclude]
  #image_3D.set_shape([256, 256, 3])
    image_3D = image_3D.astype('float32')

    return image_3D
def progressbar(progress,total):
    percent= 100*(progress/ float(total))
    bar= '=' * int(percent) + '-' * (100- int(percent))
    print(f"\r |{bar}| {percent:.2f}%", end="\r")
    
def visualize_3D(X):
    """
    Visualize the image middle slices for each axis
    """
    a,b,c = X.shape
    
    plt.figure(figsize=(15,15))
    plt.subplot(131)
    plt.imshow(np.rot90(X[a//2, :, :]),vmin=-1, vmax=1, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(np.rot90(X[:, b//2, :]),vmin=-1, vmax=1, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(X[:, :, c//2],vmin=-1, vmax=1, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

def visualize_2D(X,y):
    """
    Visualize the image first slice
    """    
    plt.figure(figsize=(15,15))
    plt.subplot(121)
    plt.imshow(X[:, :, 0],vmin=-1, vmax=1, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Input')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(y[:, :, 0],vmin=-1, vmax=1, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Target')
    plt.axis('off')

    
def normalize(X, mode='Z-score'):
  Xminval=X.min()
  Xmaxval=X.max()
  Xdyrange=Xmaxval-Xminval
  mean_X=np.mean(X)
  std_X=np.std(X)
  if mode=='Z-score':
     X=(X - mean_X) / std_X
  if mode=='-1to1':
    X=X/(Xdyrange/2)-1

  if mode=='0-1':
    X=X/(Xdyrange/2)-1
  return X

def normalise_per_channel(image):
    rank = 4
    ch_mins = np.amin(image, axis=tuple(range(rank - 1)))
    ch_maxs = np.amax(image, axis=tuple(range(rank - 1)))
    ch_range = ch_maxs - ch_mins
    
    #idx = np.where(ch_range == 0)
    #ch_mins[idx] = 0
    #ch_range[idx] = 1
    img = (image - ch_mins) / ch_range
    image_norm = 2*img - 1
    
    return image_norm


def normalize_tensor(input_image, real_image):
    input_image=normalise_per_channel(input_image)
    real_image=normalise_per_channel(real_image)

    return input_image, real_image

def normalize_per_slice(image):
    image_norm = np.zeros_like(image)
    #print(image.shape)
    w, h, c= image.shape
    for i in range(c):
        im_range=np.max(image[...,i])-np.min(image[...,i])
        img=(image[...,i]-np.min(image[...,i]))/im_range                                
        image_norm[...,i] = 2*img - 1
    
    return image_norm

def prepare_data(data_dir, X_folder, Y_folder,model_dim,im_size,SubList):
    j=0
    total=len(os.listdir(os.path.join(data_dir,'all_NG2_low')))/2
    progressbar(0, total)
    X_input=[]
    y_target=[]
    X_input3D=[]
    y_target3D=[] 
    for fileID in os.listdir(os.path.join(data_dir,X_folder)):
        
        name, ext = os.path.splitext(fileID)
        file_idX = name.split("_")
        if file_idX[0] in SubList:
            file_idy=file_idX[0]+'_00047.img'
            if ext == '.img':
                #print(fileID)
                PET_X=nifty_imageloader(os.path.join(data_dir,X_folder,fileID))
                PET_y=nifty_imageloader(os.path.join(data_dir,Y_folder,file_idy))
                pl_x=PET_X[:,:,-1] ##  last slice
                pl_y=PET_y[:,:,-1] ##  last slice
                PET_X=np.dstack((PET_X,pl_x)) #extending dimension by adding the last slice to the volume 
                PET_y=np.dstack((PET_y,pl_y)) #extending dimension by adding the last slice to the volume 
                progressbar(j+1, total)
                j=j+1
                #print(j)
                if model_dim=='3D':
                    X_img=normalize_per_slice(PET_X)
                    y_img=normalize_per_slice(PET_y)

                    X_img =np.rot90(resize(X_img, (im_size, im_size))) #rotate for better visualization
                    y_img =np.rot90(resize(y_img, (im_size, im_size)))
                    X_img = tf.cast(X_img, tf.float32)              
                    y_img = tf.cast(y_img, tf.float32)
                    X_input3D.append(X_img)
                    y_target3D.append(y_img)
                    X_input=np.expand_dims(np.array(X_input3D), axis=-1)
                    y_target=np.expand_dims(np.array(y_target3D), axis=-1)
                    
                elif model_dim=='2D':
          
                    for plane in range(PET_X.shape[-1]):  ## slice by slice append
                        X_img = PET_X[:, :, plane]
                        y_img = PET_y[:, :, plane]
                        y_img=np.expand_dims(y_img, axis=-1)
                        X_img=normalize(X_img,'-1to1')## -1to1 or 0-1 or Z-score
                        y_img=normalize(y_img,'-1to1')
                        
                        X_img =np.rot90(resize(X_img, (im_size, im_size))) #rotate for better visualization
                        y_img =np.rot90(resize(y_img, (im_size, im_size)))

                        #X_img,real_image=normalize_tensor(X_img,y_img)
                        X_img = tf.cast(X_img, tf.float32)              
                        y_img = tf.cast(y_img, tf.float32)

                        X_input.append(X_img)  
                        y_target.append(y_img)
                               
        
                elif model_dim=='2.5D':
                    
                     for plane in range(PET_X.shape[-1]):
                        if plane == 0:
                            X_img = np.concatenate([PET_X[:, :, :1], PET_X[:, :, :2]], axis=-1)
                        # end edge
                        elif plane == PET_X.shape[-1] - 1:
                            X_img = np.concatenate([PET_X[:, :, -1:], PET_X[:, :, -2:]], axis=-1)
                        else:
                            X_img = PET_X[:, :, plane - 1:plane + 2]
                            
                        y_img=PET_y[:,:,plane]  ##if we want to pair 3 input slices to 3 target slices
                        y_img=np.expand_dims(y_img, axis=-1)
                        X_img=normalize_per_slice(X_img)
                        y_img=normalize_per_slice(y_img)
                        #X_img=normalize(X_img,'-1to1')## -1to1 or 0-1 or Z-score
                        #y_img=normalize(y_img,'-1to1')
                        
                        X_img =np.rot90(resize(X_img, (im_size, im_size))) #rotate for better visualization
                        y_img =np.rot90(resize(y_img, (im_size, im_size)))

                        #X_img,real_image=normalize_tensor(X_img,y_img)
                        X_img = tf.cast(X_img, tf.float32)              
                        y_img = tf.cast(y_img, tf.float32)

                        X_input.append(X_img)  
                        y_target.append(y_img)
        
                else:
                  raise TypeError('Image dimension not defined.')

                del(PET_X, PET_y)
                
        
        else:
          #print('skipping this subject:',file_idX[0])
            continue
        
       
    return   X_input,y_target
        

def generate_images(model, input_image, target):
    prediction = model(input_image, training=True)
    plt.figure(figsize=(15, 15))
    
    #a,b,c,d,e=input_image.shape()
    if input_image.shape[3] >3:
        imslice=18
        diff_image=np.abs(target[0,:,:,imslice,0]-prediction[0,:,:,imslice,0])
        display_list = [input_image[0,:,:,imslice,0], target[0,:,:,imslice,0], prediction[0,:,:,imslice,0],diff_image]
    else:
        imslice=0
        diff_image=np.abs(target[0,:,:,imslice]-prediction[0,:,:,imslice])
        display_list = [input_image[0,:,:,imslice], target[0,:,:,imslice], prediction[0,:,:,imslice],diff_image]

    
    title = ['Input Image', 'Target', 'Synthetic Image', 'Residual']

    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        if i<3:
            colormap='hot'
        else:
            colormap='bwr'
            
        plt.imshow(display_list[i] * 0.5 + 0.5, cmap=colormap)
        plt.axis('off')
        #plt.savefig('./test output per sample/saved_prediction_'+ str(i) + '.png', dpi=300)
    plt.show()


def save_nifty(img, filename, denorm=False):
    """ To convert a numpy volume to nifty """

    # TODO fix the header

    # reshape
    img = img.squeeze().swapaxes(1, 0).swapaxes(2, 1)

    # de-normalize images
    if denorm:
        mean = denorm[0]
        std = denorm[1]
        img = (img * std) + mean

    # template
    template = 'data/template.img'
    template_img = nib.load(template)
    template_hdr = template_img.header
    # template_arr = template_img.get_data()
    template_affine = template_img.affine  # voxel_dim = [1.37, 1.37, 3.27]  # x, y, z (mm)

    # numpy to nifty
    img_nifty = nib.Nifti1Image(img, affine=template_affine, header=template_hdr)

    # save nifty
    nib.save(img_nifty, filename)

def denorm(y,yhat):
    dyrang=np.max(y)-np.min(y)
    return dyrang*(yhat+1)/2

def denorm_perslice(y,yhat):
    image_denorm = np.zeros_like(y)
    #print(image.shape)
    w, h, c= y.shape
    for i in range(c):
        im_range=np.max(y[...,i])-np.min(y[...,i])
        image_denorm[...,i] = im_range*(yhat[...,i]+1)/2

    return image_denorm