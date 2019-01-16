# import tensorflow as tf
import torch
import numpy as np
import os
import time

import cv2

# import vgg.Vgg16 as Vgg16
from vgg.vgg import Vgg16
import scipy.misc as scm

# slim = tf.contrib.slim

mse_loss = torch.nn.MSELoss(reduce=False, size_average=False)

class Differ():
    def __init__(self, img_w, img_h, is_cuda=True):
        # self.sess = tf.Session()
        
        self.img_w = img_w
        self.img_h = img_h
        
        # self.build_vgg()

        self.device = torch.device("cuda" if is_cuda else "cpu")

        self.vgg = Vgg16(requires_grad=False).to(self.device)

    # def build_vgg(self):
    #     self.canvas = tf.placeholder(tf.float32, [1, None, None, 3])
    #     self.target = tf.placeholder(tf.float32, [1, None, None, 3])
        
    #     with slim.arg_scope(vgg.vgg_arg_scope()):
    
    #         f1, f2, f3, f4, exclude = vgg.vgg_16(tf.concat([self.canvas, self.target], axis=0))
    
    #         canvas_f, target_f = tf.split(f3, 2, 0)
    
    #         # load vgg model
    #         vgg_model_path = "vgg/vgg_16.ckpt"
    #         vgg_vars = slim.get_variables_to_restore(include=['vgg_16'], exclude=exclude)
    #         # vgg_init_var = slim.get_variables_to_restore(include=['vgg_16/fc6'])
    #         init_fn = slim.assign_from_checkpoint_fn(vgg_model_path, vgg_vars)
    #         init_fn(self.sess)
    #         # tf.initialize_variables(var_list=vgg_init_var)
    #         print('vgg s weights load done')
        
    #     self.canvas_f = canvas_f
    #     self.target_f = target_f

    def positive_sharpen(self,i,overblur=False,coeff=8.): #no darken to original image
        # emphasize the edges
        #print(i.shape)
        blurred = cv2.blur(i,(5,5))
        sharpened = i + (i - blurred) * coeff
        if overblur:
            return cv2.blur(np.maximum(sharpened,i),(11,11))
        return cv2.blur(np.maximum(sharpened,i),(3,3))

    def diff(self,i1,i2,overblur=False):
        # # use rgb
        d = (i1-i2)# * [0.2,1.5,1.3]
        d = d*d

        d = self.positive_sharpen(np.sum(d,-1), overblur=overblur)
        return d
        # grayscalize

    def diff_vgg(self, i1, i2, overblur=False):
        # # use rgb
        #i1_tensor = tf.convert_to_tensor(i1)
        #i2_tensor = tf.convert_to_tensor(i2)
        i1_in = np.expand_dims(i1,0)
        i2_in = np.expand_dims(i2,0)

        i1_tensor = torch.tensor(i1_in).permute(0,3,1,2)
        i2_tensor = torch.tensor(i2_in).permute(0,3,1,2)

        i1_tensor = i1_tensor.to(self.device)
        i2_tensor = i2_tensor.to(self.device)
        # y = transformer(x)

        # y = utils.normalize_batch(y)
        # x = utils.normalize_batch(x)

        features_i1 = self.vgg(i1_tensor)
        features_i2 = self.vgg(i2_tensor)
        
        content_loss = mse_loss(features_i1.relu2_2, features_i2.relu2_2).mean(1).squeeze().cpu().numpy()
        # content_loss = torch.

        # print("content_loss shape: ",content_loss.shape)
            
        d = self.positive_sharpen(content_loss,overblur=overblur)
        
        return d

    def diff_vgg_batch(self, i1, i2, overblur=False):
        # # use rgb
        #i1_tensor = tf.convert_to_tensor(i1)
        #i2_tensor = tf.convert_to_tensor(i2)
        # i1_in = np.expand_dims(i1,0)
        # i2_in = np.expand_dims(i2,0)

        i1_tensor = torch.tensor(i1).permute(0,3,1,2)
        i2_tensor = torch.tensor(i2).permute(0,3,1,2)

        i1_tensor = i1_tensor.to(self.device)
        i2_tensor = i2_tensor.to(self.device)
        # y = transformer(x)

        # y = utils.normalize_batch(y)
        # x = utils.normalize_batch(x)

        features_i1 = self.vgg(i1_tensor)
        features_i2 = self.vgg(i2_tensor)
        
        content_loss = mse_loss(features_i1.relu2_2, features_i2.relu2_2).mean(1).cpu().numpy()
        # content_loss = torch.

        # print("content_loss shape: ",content_loss.shape)
            
        # d = self.positive_sharpen(content_loss,overblur=overblur)
        
        return content_loss

    def new_diff(self, i1=None, i2=None, alpha=0.4):
        dd = self.diff_vgg(i1,i2,overblur=True)
        
        dd_w = i1.shape[0]
        dd_h = i1.shape[1]
        dd_r = cv2.resize(dd,(dd_h, dd_w))
        
        # find out where max difference point is.
        d = self.diff(i1,i2,overblur=True)
        
        d_mean = dd_r*alpha + d*(1.0-alpha)

        return d_mean

    def wherediff(self, i1=None, i2=None, alpha=0.4):
        dd = self.diff_vgg(i1,i2,overblur=True)
        
        dd_w = i1.shape[0]
        dd_h = i1.shape[1]
        dd_r = cv2.resize(dd,(dd_h, dd_w))
        
        # find out where max difference point is.
        d = self.diff(i1,i2,overblur=True)
        
        d_mean = dd_r*alpha + d*(1.0-alpha)
        
        #print("d shape: ", d.shape)
        #print("dd shape: ", dd.shape) 
        #print("dd r shape: ", dd_r.shape)
    
        i,j = np.unravel_index(d_mean.argmax(),d_mean.shape)
        return i,j,d_mean
