#!/usr/bin/env python
# coding: utf-8


print("\n * START ENVIRONMENT CONFIGURATION* \n")
import os
#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
os.chdir('/home/mojjaf/Pix2Pix_CardiacPET')

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[1:2], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
print("\n * Environment and GPU Setup compelted... * \n")

print("Available devices on Hoffman Stanford:")
for i, device in enumerate(tf.config.list_logical_devices()):
  print("%d) %s" % (i, device))



import numpy as np
import pandas as pd
from tensorflow import keras
import time
import datetime
import pandas as pd
from sys import stdout
from model_args import get_args
from models import Generator3D, Discriminator3D
from utils import prepare_data
from losses import generator_loss, discriminator_loss

print("\n * Python libraries loaded successfully... * \n")


dataset= "cardiac_PET_DenoisingGAN"
mode='3D'  #Image volumes are in 3D format
experiment = "/experiments/Pix2Pix_"+dataset+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"  # 
model_name = "model_3D_"+dataset+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/" 
print(f"\nExperiment: {experiment}\n")
args = get_args()
project_dir = args.main_dir

experiment_dir = project_dir+experiment+mode

if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir) 
    
checkpoint_dir = experiment_dir+model_name
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir) 
    
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

print("\n * Patient list loaded and data subsets created... * \n")



print("\n * Loading nifty images... (this might take a while) * \n")
start_time = time.time()
data_dir=args.data_dir
with tf.device('/device:cpu:0'):
    Xtrain,ytrain=prepare_data(data_dir, 'all_NG2_low', 'all_NG2',model_dim='3D',im_size=args.image_size,SubList=train_subjects)
    print("\n * Training data loaded.DONE * \n")
    Xval,yval=prepare_data(data_dir, 'all_NG2_low', 'all_NG2',model_dim='3D',im_size=args.image_size,SubList=val_subjects)
    print("\n * Validation data loaded. DONE * \n")
    

print("Total number of training samples ", len(Xtrain))
print("Total number of validation samples ",len(Xval))


data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
])

augment=False
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE=(args.batch_size)
with tf.device('/device:cpu:0'):
    datasetx = tf.data.Dataset.from_tensor_slices(Xtrain)

    datasety = tf.data.Dataset.from_tensor_slices(ytrain)
    train_dataset = tf.data.Dataset.zip((datasetx, datasety))
    if augment:
        train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

with tf.device('/device:cpu:0'):
    datasetx = tf.data.Dataset.from_tensor_slices(Xval)

    datasety = tf.data.Dataset.from_tensor_slices(yval)
    valid_dataset = tf.data.Dataset.zip((datasetx, datasety))
    val_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = valid_dataset.batch(BATCH_SIZE)
    
print("\n * Train and validation zipped tensor pairs ready.DONE * \n")
print("\n * Preprocessing: DONE * \n")
print("--- total processing time: %s seconds ---" % (time.time() - start_time))

# Models
print("\n * Genarator and Discriminaror models loaded. Check model summaries * \n")
generator = Generator3D()
discriminator = Discriminator3D()
generator.summary()
discriminator.summary()


# Models
G = Generator3D()
D = Discriminator3D()

# Optimizers
lr=2e-4

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=lr,
    decay_steps=8000,
    decay_rate=0.9,staircase=False)

generator_optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)
@tf.function
def train_step(image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        gen_output = G(image, training=True)
        disc_real_output = D([image, target], training=True)
        disc_fake_output = D([image, gen_output], training=True)
        disc_loss = discriminator_loss(disc_real_output, disc_fake_output)
        
        total_gan_loss, gen_gan_loss, l1_loss = generator_loss(target, gen_output, disc_fake_output)

    generator_gradients = gen_tape.gradient(total_gan_loss, G.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, D.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, G.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, D.trainable_variables))
        
    return total_gan_loss, l1_loss, gen_gan_loss
        
@tf.function
def test_step(image, target):
    gen_output = G(image, training=False)

    disc_real_output = D([image, target], training=False)
    disc_fake_output = D([image, gen_output], training=False)
    disc_loss = discriminator_loss(disc_real_output, disc_fake_output)

    total_gan_loss, gen_gan_loss, l1_loss = generator_loss(target, gen_output, disc_fake_output)
        
    return total_gan_loss, l1_loss, gen_gan_loss

def fit(train_gen, valid_gen, epochs):
    
    path = checkpoint_dir 
    if os.path.exists(path)==False:
        os.mkdir(path)
        
    Nt = len(train_gen)
    history = {'train': [], 'valid': []}
    prev_loss = np.inf
    
    epoch_Gen3D_loss = tf.keras.metrics.Mean()
    epoch_L1_loss = tf.keras.metrics.Mean()
    epoch_disc_loss = tf.keras.metrics.Mean()
    epoch_Gen3D_loss_val = tf.keras.metrics.Mean()
    epoch_L1_loss_val = tf.keras.metrics.Mean()
    epoch_disc_loss_val = tf.keras.metrics.Mean()
    
    for e in range(epochs):
        print('Epoch {}/{}'.format(e+1,epochs))
        b = 0
        for Xb, yb in train_gen:
            b += 1
            losses = train_step(Xb, yb)
            epoch_Gen3D_loss.update_state(losses[0])
            epoch_L1_loss.update_state(losses[1])
            epoch_disc_loss.update_state(losses[2])
            
            stdout.write('\rBatch: {}/{} - gen_loss: {:.4f} - L1_loss: {:.4f} - disc_loss: {:.4f}'
                         .format(b, Nt, epoch_Gen3D_loss.result(), epoch_L1_loss.result(), epoch_disc_loss.result()))
            stdout.flush()
        history['train'].append([epoch_Gen3D_loss.result(), epoch_L1_loss.result(), epoch_disc_loss.result()])
        
        for Xb, yb in valid_gen:
            losses_val = test_step(Xb, yb)
            epoch_Gen3D_loss_val.update_state(losses_val[0])
            epoch_L1_loss_val.update_state(losses_val[1])
            epoch_disc_loss_val.update_state(losses_val[2])
            
        stdout.write('\n               gen_loss_val: {:.4f} - L1_loss_val: {:.4f} - disc_loss_val: {:.4f}'
                     .format(epoch_Gen3D_loss_val.result(), epoch_L1_loss_val.result(), epoch_disc_loss_val.result()))
        stdout.flush()
        history['valid'].append([epoch_Gen3D_loss_val.result(), epoch_L1_loss_val.result(), epoch_disc_loss_val.result()])
        
        y_pred = G.predict(Xb)
        y_true = yb
        
        idx = np.random.randint(len(Xb))
    
        
        # save models
        print(' ')
        G.save_weights(path + '/all_epochs_Generator3D.h5') 
        if epoch_Gen3D_loss_val.result() < prev_loss:    
            G.save_weights(path + '/Generator3D.h5') 
            D.save_weights(path + '/Discriminator3D.h5')
            print("Validation loss decresaed from {:.4f} to {:.4f}. Models' weights are now saved.".format(prev_loss, epoch_Gen3D_loss_val.result()))
            prev_loss = epoch_Gen3D_loss_val.result()
        else:
            print("Validation loss did not decrese from {:.4f}.".format(prev_loss))
        print(' ')
        
        # resets losses states
        epoch_Gen3D_loss.reset_states()
        epoch_L1_loss.reset_states()
        epoch_disc_loss.reset_states()
        epoch_Gen3D_loss_val.reset_states()
        epoch_L1_loss_val.reset_states()
        epoch_disc_loss_val.reset_states()
        
        del Xb, yb, y_pred, y_true, idx
        
    return history

print("\n * Training configuration ready. DONE * \n")


print("\n * Now Training... (this might take a while) * \n")

fit(train_dataset, val_dataset, args.nb_epochs)#

print("\n * Model training is completed. DONE * \n")
