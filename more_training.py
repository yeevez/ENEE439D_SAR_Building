import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.layers import Input, Dense, GlobalAveragePooling1D, Reshape, Conv2D, concatenate
import sys
sys.path.append('swin_transformer_repo/keras-vision-transformer')

from keras_vision_transformer import swin_layers
from keras_vision_transformer import transformer_layers

def accuracy(y_true,y_pred):
  accuracy = K.sum(tf.cast((y_true==y_pred),tf.float32))/tf.cast(K.prod(K.shape(y_true)),tf.float32)
  return accuracy

def IoU(y_true,y_pred):
  true_pos = K.sum(K.round(K.clip(y_true*y_pred,0,1)))
  #print("True pos: ", true_pos)
  total_pos = K.sum(K.round(K.clip(y_pred,0,1)))
  #print("total pos: ", total_pos)
  ground_pos = K.sum(y_true)
  #print("ground truth positive:",ground_pos)
  false_pos = total_pos - true_pos
  #print("false pos", false_pos)
  false_neg = ground_pos - true_pos
  #print("false neg", false_neg)
  IoU = (true_pos)/(true_pos + false_pos + false_neg + K.epsilon())
  return IoU

def precision(y_true,y_pred):
  true_pos = K.sum(K.round(K.clip(y_true*y_pred,0,1)))
  total_pos = K.sum(K.round(K.clip(y_pred,0,1)))
  precision = true_pos/(total_pos + K.epsilon())
  return precision

def recall(y_true,y_pred):
  true_pos = K.sum(K.round(K.clip(tf.cast(y_true,tf.float32)*y_pred,0,1)))
  ground_pos = K.sum(y_true)
  recall = true_pos/(ground_pos + K.epsilon())
  return recall

def F1_score(y_true,y_pred):
  prec = precision(y_true,y_pred)
  rec = recall(y_true,y_pred)
  F1_score = (2*prec*rec)/(prec+rec + K.epsilon())
  return F1_score

"""##Loss Functions

See https://arxiv.org/abs/2006.14822
"""

def class_weight(labels):
  building_ratio = sum(sum(sum(labels)))/(labels.shape[0]*256*256)
  return building_ratio


def dice_coef(y_true, y_pred):
  smooth = 1
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(tf.cast(y_true_f,tf.float32)* y_pred_f)
  return (2. * intersection + smooth) / (K.sum(tf.cast(y_true_f,tf.float32)) + K.sum(y_pred_f) + smooth)
    
    
def dice_coef_loss(y_true, y_pred):
  return 1 - dice_coef(y_true, y_pred)

def compound_loss(y_true,y_pred):
  return 0.9*dice_coef_loss(y_true,y_pred)+ 0.1*K.binary_crossentropy(y_true, y_pred)


def weighted_bce(y_true,y_pred):
  num_pred = K.sum(K.cast(y_pred < 0.5, y_true.dtype)) + K.sum(y_true)    
  zero_weight =  K.sum(y_true)/ num_pred +  K.epsilon() 
  one_weight = K.sum(K.cast(y_pred < 0.5, y_true.dtype)) / num_pred +  K.epsilon()
  weights =  (1.0 - y_true) * zero_weight +  y_true * one_weight 
  bin_crossentropy = K.binary_crossentropy(y_true, y_pred)
  weighted_bin_crossentropy = weights * bin_crossentropy 
  return K.mean(weighted_bin_crossentropy)

img_train = np.load("processed_data/img_train.npy")/255.0
img_test = np.load("processed_data/img_test.npy")/255.0
label_train = np.load("processed_data/label_train.npy").astype(np.float32)
label_test = np.load("processed_data/label_test.npy").astype(np.float32)

custom_objects = {"precision": precision, "recall": recall, "F1_score": F1_score, "compound_loss": compound_loss, "dice_coef": dice_coef, "dice_coef_loss": dice_coef_loss}

unet = tf.keras.models.load_model('unet_model', custom_objects = custom_objects)

checkpoint_filepath = '/tmp/checkpoint'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_binary_io_u',
    mode='max',
    save_best_only=True)

unet.fit(img_train,label_train, batch_size=128, epochs=5,validation_data=(img_test,label_test),callbacks=[model_checkpoint_callback])

unet.load_weights(checkpoint_filepath)

unet.save('unet_model')

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_binary_io_u',
    mode='max',
    save_best_only=True)

transformer = tf.keras.models.load_model('swin_transformer_model', custom_objects = custom_objects)

transformer.fit(img_train,label_train, batch_size=128, epochs=5,validation_data=(img_test,label_test),callbacks=[model_checkpoint_callback])

transformer.load_weights(checkpoint_filepath)

transformer.save('swin_transformer_model')



