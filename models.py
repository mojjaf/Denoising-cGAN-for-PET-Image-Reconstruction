import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Conv3D, Conv3DTranspose, Dropout, ReLU, LeakyReLU, Concatenate, ZeroPadding3D
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

from main_arguments import get_args

args = get_args()

IMG_WIDTH = args.image_size
IMG_HEIGHT = args.image_size
INPUT_CHANNELS=args.input_channel
OUTPUT_CHANNELS=args.output_channel
CHANNELS=args.input_channel

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result



def Generator2D():
    inputs = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS])
    filter=64

    down_stack = [
    downsample(filter, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(filter*2, 4),  # (batch_size, 64, 64, 128)
    downsample(filter*4, 4),  # (batch_size, 32, 32, 256)
    downsample(filter*8, 4),  # (batch_size, 16, 16, 512)
    downsample(filter*8, 4),  # (batch_size, 8, 8, 512)
    downsample(filter*8, 4),  # (batch_size, 4, 4, 512)
    downsample(filter*8, 4),  # (batch_size, 2, 2, 512)
    downsample(filter*8, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
    upsample(filter*8, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(filter*8, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(filter*8, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(filter*8, 4),  # (batch_size, 16, 16, 1024)
    upsample(filter*4, 4),  # (batch_size, 32, 32, 512)
    upsample(filter*2, 4),  # (batch_size, 64, 64, 256)
    upsample(filter, 4),  # (batch_size, 128, 128, 128)
    ]
    
  
    

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

     # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
def Discriminator2D():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
    kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


   
def Generator3D():
    '''
    Generator model
    '''
    def encoder_step(layer, Nf, ks, norm=True):
        x = Conv3D(Nf, kernel_size=ks, strides=2, kernel_initializer='he_normal', padding='same')(layer)
        if norm:
            x = InstanceNormalization()(x)
            #x=tf.keras.layers.BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        return x

    def bottlenek(layer, Nf, ks):
        x = Conv3D(Nf, kernel_size=ks, strides=2, kernel_initializer='he_normal', padding='same')(layer)
        x = InstanceNormalization()(x)
        x = LeakyReLU()(x)
        for i in range(4):
            y = Conv3D(Nf, kernel_size=ks, strides=1, kernel_initializer='he_normal', padding='same')(x)
            x = InstanceNormalization()(y)
            #x= tf.keras.layers.BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = Concatenate()([x, y])

        return x

    def decoder_step(layer, layer_to_concatenate, Nf, ks):
        x = Conv3DTranspose(Nf, kernel_size=ks, strides=2, padding='same', kernel_initializer='he_normal')(layer)
        x = InstanceNormalization()(x)
        #x=tf.keras.layers.BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Concatenate()([x, layer_to_concatenate])
        x = Dropout(0.2)(x)
        return x

    layers_to_concatenate = []
    inputs = Input((IMG_WIDTH,IMG_HEIGHT,CHANNELS,1), name='input_image')
    Nfilter_start = 64
    depth = 3
    ks = 4
    x = inputs

    # encoder
    for d in range(depth-1):
        if d==0:
            x = encoder_step(x, Nfilter_start*np.power(2,d), ks, False)
        else:
            x = encoder_step(x, Nfilter_start*np.power(2,d), ks)
        layers_to_concatenate.append(x)

    # bottlenek
    x = bottlenek(x, Nfilter_start*np.power(2,depth-1), ks)

    # decoder
    for d in range(depth-2, -1, -1): 
        x = decoder_step(x, layers_to_concatenate.pop(), Nfilter_start*np.power(2,d), ks)

    # classifier
    last = Conv3DTranspose(1, kernel_size=ks, strides=2, padding='same', kernel_initializer='he_normal', activation='tanh', name='output_generator')(x)
    #print(tf.shape(last))
    #last=
    return Model(inputs=inputs, outputs=last, name='Generator')

def Discriminator3D():
    '''
    Discriminator model
    '''

    inputs = Input((IMG_WIDTH,IMG_HEIGHT,CHANNELS,1), name='input_image')
    targets = Input((IMG_WIDTH,IMG_HEIGHT,CHANNELS,1), name='target_image')
    Nfilter_start = 64
    depth = 3
    ks = 4
    

    def encoder_step(layer, Nf, norm=True):
        x = Conv3D(Nf, kernel_size=ks, strides=2, kernel_initializer='he_normal', padding='same')(layer)
        if norm:
            x = InstanceNormalization()(x)
            #x=tf.keras.layers.BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        
        return x

    x = Concatenate()([inputs, targets])

    for d in range(depth):
        if d==0:
            x = encoder_step(x, Nfilter_start*np.power(2,d), False)
        else:
            x = encoder_step(x, Nfilter_start*np.power(2,d))
            
    x = ZeroPadding3D()(x)
    x = Conv3D(Nfilter_start*(2**depth), ks, strides=1, padding='valid', kernel_initializer='he_normal')(x) 
    x = InstanceNormalization()(x)
    x = LeakyReLU()(x)
      
    x = ZeroPadding3D()(x)
    last = Conv3D(1, ks, strides=1, padding='valid', kernel_initializer='he_normal', name='output_discriminator')(x) 

    return Model(inputs=[inputs, targets], outputs=last, name='Discriminator')

