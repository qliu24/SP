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
            self.model_weights = tf.placeholder(tf.float32, [self.patch_len, self.patch_len, self.vc_num, self.sp_num])
            self.model_logZs = tf.placeholder(tf.float32, [self.sp_num])
            self.model_logPriors = tf.placeholder(tf.float32, [self.sp_num])
            
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
        
        