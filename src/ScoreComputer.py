import tensorflow as tf
import numpy as np
import cv2
import tensorflow.contrib.slim as slim
from datetime import datetime
import os

class ScoreComputer:
    def __init__(self, patch_r, vc_num, sp_num):
        self.patch_len = patch_r*2+1
        self.vc_num = vc_num
        self.sp_num = sp_num
        
        with tf.device('/cpu:0'):
            self.patch_feat = tf.placeholder(tf.float32, [1, None, None, self.vc_num])
            self.model_weights = tf.placeholder(tf.float32, [self.patch_len, self.patch_len, self.vc_num, None])
            self.model_logZs = tf.placeholder(tf.float32, [None])
            self.model_logPriors = tf.placeholder(tf.float32, [None])
            
        self.scores = tf.nn.conv2d(self.patch_feat, self.model_weights, strides=[1,1,1,1], padding='SAME')
        self.scores = self.scores - self.model_logZs + self.model_logPriors
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        
    def comptScore(self, patch_feat, sp_models):
        weights = np.array([mm[0] for mm in sp_models]).reshape(self.sp_num, self.patch_len, self.patch_len, self.vc_num)
        weights = np.transpose(weights, [1,2,3,0])
        logZs = np.array([mm[1] for mm in sp_models])
        priors = np.array([mm[2] for mm in sp_models])
        
        result = self.sess.run(self.scores, feed_dict={self.patch_feat: [patch_feat], \
                                                      self.model_weights: weights, \
                                                      self.model_logZs: logZs, \
                                                      self.model_logPriors: priors})
        
        return result[0]
    
    
    def comptScore_mixture(self, patch_feat, sp_models, cls_num):
        # first sp_num-2 models are mixture models for SP
        # last 2 models are unary model for BG
        sp_num = self.sp_num-2
        sp_num_bg = 2
        assert(len(sp_models)==self.sp_num)
        
        weights = np.array([[mm[kk][0] for kk in range(cls_num)] for mm in sp_models[0:-2]]).reshape(sp_num*cls_num, self.patch_len, self.patch_len, self.vc_num)
        weights_bg = np.array([mm[0] for mm in sp_models[-2:]]).reshape(sp_num_bg, self.patch_len, self.patch_len, self.vc_num)
        
        weights = np.transpose(np.concatenate((weights,weights_bg), axis=0), [1,2,3,0])
        
        logZs = np.array([[mm[kk][1] for kk in range(cls_num)] for mm in sp_models[0:-2]]).ravel()
        logZs_bg = np.array([mm[1] for mm in sp_models[-2:]])
        logZs = np.append(logZs, logZs_bg)
        
        priors_kk = np.array([[mm[kk][2] for kk in range(cls_num)] for mm in sp_models[0:-2]]).ravel()
        for ppbg in range(sp_num_bg):
            priors_kk = np.append(priors_kk, 0.0)
        
        priors = np.array([[mm[kk][3] for kk in range(cls_num)] for mm in sp_models[0:-2]]).ravel()
        priors_bg = np.array([mm[2] for mm in sp_models[-2:]])
        priors = np.append(priors, priors_bg)
        
        result = self.sess.run(self.scores, feed_dict={self.patch_feat: [patch_feat], \
                                                      self.model_weights: weights, \
                                                      self.model_logZs: logZs, \
                                                      self.model_logPriors: priors_kk+priors})[0]
        
        result_final = np.zeros((result.shape[0], result.shape[1], self.sp_num))
        for pp in range(sp_num):
            result_final[:,:,pp] = np.max(result[:,:,(pp*4):(pp+1)*4], axis=-1)
            
        for ppbg in range(sp_num_bg):
            result_final[:,:,sp_num+ppbg] = result[:,:,4*sp_num+ppbg]\
            
        return result_final
        
        