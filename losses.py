
import tensorflow as tf

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True) 
LAMBDA = 10

def discriminator_loss(disc_real_output, disc_fake_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    fake_loss = loss_object(tf.zeros_like(disc_fake_output), disc_fake_output)
    disc_loss = real_loss + fake_loss
    
    return disc_loss


def generator_loss(target, gen_output, disc_fake_output):

    gan_loss = loss_object(tf.ones_like(disc_fake_output), disc_fake_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
       
    # total loss
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)   

    return total_gen_loss,gan_loss, l1_loss
