# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:09:52 2017

@author: abhishek
"""
# import tensorflow as tf
import theano.tensor as T
# Import utilities
# from utilities import *
# from config import *

class NormalizingPlanarFlow(object):

    def __init__(self,
                 z, # shape (nSamples, nDim)
                 z_dim=2):
        self.z=z
        self.z_dim= z_dim
            
#        self.w = tf.Variable(xavier_init(self.z_dim, 1), name = "w_planar_flow")
#        self.u = tf.Variable(xavier_init(self.z_dim, 1), name = "u_planar_flow")
#        self.b = tf.Variable(tf.zeros(shape=[]), "b_planar_flow")

    def softplus(self, x):
            return T.log(T.clip(x, 1e-4, 0.98))

    def tanh(self, x):
        return T.tanh(x)

    def dtanh(self, tensor):
        return 1.0 - T.square(T.tanh(tensor))

    def tf_norm(self, x):
        return T.sqrt(T.sum(x ** 2, 1))

    def planar_flow(self, z, flow_params, K= 5, Z= 100,
                      invert_condition = True):
        """
        THIS IS THE ONE WHICH IS BEING USED CURRENTLY.
        Same as the previous implementation with the constraints. Except that in
        this case, flow is handled in a for loop and that it returns the sum of 
        logdet_jacobians. 
        z: z0 as outputted by the encoder
        lambd: Flow params
        K: nFlows
        Z: z_dim
        """
        us, ws, bs = flow_params
    
        log_detjs = []
        if K == 0:
            # f_z = z
            sum_logdet_jacobian = logdet_jacobian = 0
        else:    
            for k in range(K):
                u, w, b = us[:, k*Z:(k+1)*Z], ws[:, k*Z:(k+1)*Z], bs[:, k]
                # print "u shape", u.get_shape()
                # print "w shape", w.get_shape()
                # print "z shape", self.z.get_shape()
                # print "b shape", b.get_shape()
                if invert_condition:
                    uw = T.sum(T.dot(w, u.T),axis=1,keepdims=True) # u: (?,2), w: (?,2), b: (?,)



                    uw = T.cast(uw, 'float32')
                    muw = -1 + self.softplus(uw)
                    
                    norm_w = self.tf_norm(w)
                    # print "norm_w shape", norm_w.get_shape()
                    inverse_norm_w = 1/ norm_w
                    # print "muw shape", muw.get_shape()
                    # print "uw shape", uw.get_shape()
                    # print "norm_w shape", norm_w.get_shape()
                    # print "reshaped norm w", tf.reshape(norm_w, shape= [-1,1]).get_shape()
                    # u_hat = u + (muw - uw) * w / norm_w
                    u_hat = u +  w / T.reshape(norm_w, [-1,1]) # u + (muw - uw) * w / norm_w # u + (muw - uw) * tf.multiply(w, inverse_norm_w) #
                else:
                    u_hat = u

                # print "u_hat shape", u_hat.get_shape()


                zw=T.sum( (T.cast(z, dtype='float32') * w),axis=1)



                # print "zw shape", zw.get_shape()
                zwb = zw + b
                # print "zwb shape", zwb.get_shape()
                # Equation 10: f(z)= z+ uh(w'z+b)
                # print "u_hat", u_hat.get_shape()
                z=  z + u_hat* T.reshape(self.tanh(zwb),[-1, 1]) # self.z is (?,2)
                psi= T.reshape((1-self.tanh(zwb)**2), [-1,1]) * w # Equation 11. # tanh(x)dx = 1 - tanh(x)**2
                # psi= tf.reduce_sum(tf.matmul(tf.transpose(1-self.tanh(zwb)**2), self.w))
                psi_u = T.sum(T.dot(u_hat, psi.T),axis=1,keepdims=True)
                # psi_u= tf.matmul(tf.transpose(u_hat), tf.transpose(psi)) # Second term in equation 12. u_transpose*psi_z
                logdet_jacobian= T.log(T.clip(T.abs_(1 + psi_u), 1e-4, 1e7)) # Equation 12
                # print "f_z shape", f_z.get_shape()
                log_detjs.append(logdet_jacobian)
                logdet_jacobian = T.concatenate(log_detjs[0:K+1], axis= 1)

                sum_logdet_jacobian = T.mean(logdet_jacobian,axis=1)
                sum_logdet_jacobian_real = T.mean(logdet_jacobian)

    
        return z, sum_logdet_jacobian,sum_logdet_jacobian_real

 