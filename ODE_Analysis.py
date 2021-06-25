""" 1)  ODE_Analysis - Simulate, solve and visualize nonlinear differential equations with probabilistic models that are optimized."""
#-----------------------------------------------------------------------------------------------------------------------------------
__author__ = "Christian Simonis"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Christian Simonis"
__email__ = "christian.Simonis.1989@gmail.com"
__status__ = "work in progress"

# Approach: The class  ODE_Analysis enables the simulation of Lotka Volterra ODEs and applying additional observation noise and offsets.
# A domain-driven time integration approach is applied, which is combined with a probabilistic correction model. 
# This model is optimized by tuning the data-driven part. The corresponding parameters are optimized via maximum marginal likelihood.
# Visualizations are provided to illustrate results.
 
#-----------------------------------------------------------------------------------------------------------------------------------
# Name                                Version                      License  
# numpy                               1.19.5                       BSD,                                         Copyright (c) 2005-2020, NumPy Developers: https://numpy.org/doc/stable/license.html#:~:text=Copyright%20(c)%202005%2D2020%2C%20NumPy%20Developers.&text=THIS%20SOFTWARE%20IS%20PROVIDED%20BY,A%20PARTICULAR%20PURPOSE%20ARE%20DISCLAIMED.
# tensorflow-probability              0.12.2                       Apache Software License , Version 2.0,       Copyright 2018 The TensorFlow Probability Authors.  All rights reserved.https://github.com/tensorflow/probability/blob/master/LICENSE
# tensorflow                          2.5.0                        Apache Software License,                     Copyright 2019 The TensorFlow Authors.  All rights reserved.https://github.com/tensorflow/tensorflow/blob/master/LICENSE                                                                                                                                                                                                                                                                                                                                                                                                                                                         
# matplotlib                          3.4.2                        Python Software Foundation License,          Copyright (c) 2002 - 2012 John Hunter, Darren Dale, Eric Firing, Michael Droettboom and the Matplotlib development team; 2012 - 2021 The Matplotlib development team: https://matplotlib.org/stable/users/license.html
# seaborn                             0.11.1                       BSD 3-Clause "New" or "Revised" License,     Copyright (c) 2012-2021, Michael L. Waskom: https://github.com/mwaskom/seaborn/blob/master/LICENSE
# scipy                               1.5.2                        BSD 3-Clause "New" or "Revised" License,     Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers: https://github.com/scipy/scipy/blob/master/LICENSE.txt
# lmfit/lmfit-py                      1.0.2                        BSD-3                                        Copyright 2021 Matthew Newville, The University of Chicag, Renee Otten, Brandeis University, Till Stensitzki, Freie Universitat Berlin, A. R. J. Nelson, Australian Nuclear Science and Technology Organisation, Antonino Ingargiola, University of California, Los Angeles, Daniel B. Allen, Johns Hopkins University, Michal Rawlik, Eidgenossische Technische Hochschule, Zurich, https://github.com/lmfit/lmfit-py/blob/master/LICENSE
# app (Lotka Volterra simulation)       -                          MIT License,                                 Copyright (c) 2018 Mostapha Kalami Heris / Yarpiz Team, https://github.com/smkalami/lotka-volterra-in-python/blob/master/app.py
# scipy/scipy-cookbook                  -                          Copyright (c) 2001, 2002 Enthought, Inc.,    All rights reserved., Copyright (c) 2003-2017 SciPy Developers., All rights reserved., https://github.com/scipy/scipy-cookbook/blob/master/LICENSE.txt
#-----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import tensorflow_probability as tfp
import tensorflow.compat.v2 as tf
import matplotlib.pyplot as plt
import seaborn as sns  
import scipy        
import lmfit                        
import random                                                      # https://docs.python.org/3/library/random.html                                            # https://docs.python.org/3/library/datetime.html
#-----------------------------------------------------------------------------------------------------------------------------------
# '''
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
#BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
#OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
#THE POSSIBILITY OF SUCH DAMAGE.
#"""
#-----------------------------------------------------------------------------------------------------------------------------------


class ODE_Analysis:
    """The purpose of the class ODE_Analysis is:
        - to describe Lotka Volterra sequences with differential equations and probabilistic models
        - to analyze and visualize results 
    """


    #Initialization
    def __init__(self):
        """initial call""" 
        
            

    
    def create_time_series(self, alpha, beta, gamma, delta, t, x0, y0, noise_factor_x, noise_factor_y, offset_x, offset_y):
        """ creates timeseries data for Lotka Volterra sequences
        e.g. --> runfile('DE.py')
        
        Input: 
                alpha: defined parameter for Lotka Volterra Model: positive, real constant
                beta:  defined parameter for Lotka Volterra Model: positive, real constant
                gamma: defined parameter for Lotka Volterra Model: positive, real constant
                delta: defined parameter for Lotka Volterra Model: positive, real constant
                t:     Time vector as input for simulation of differential equation
                x0:    initial condition for x (Preys)
                y0:    initial condition for y (Predators)
                noise_factor_x: Factor which determines the noise for x (Preys)
                noise_factor_y: Factor which determines the noise for y (Predators)
                offset_x: Offset as x player in addition to ODE: positive, real constant 
                offset_y: Offset as y player in addition to ODE: positive, real constant 
    
        Output:
                data: Simulated time series data, containing x (Preys) data and y (Predators) data

        """
        # Thanks and reference to: https://github.com/smkalami/lotka-volterra-in-python/blob/master/app.py

        #Random number generator for repeatabe results
        random.seed(0)
        
        # Dynamics of The Model
        def f(x, y):
            xdot = alpha*x - beta*x*y
            ydot = delta*x*y - gamma*y
            return xdot, ydot
        
        # State Transition using Runge-Kutta Method
        def sim(x, y, dt):
            xdot1, ydot1 = f(x, y)
            xdot2, ydot2 = f(x + xdot1*dt/2, y + ydot1*dt/2)
            xdot3, ydot3 = f(x + xdot2*dt/2, y + ydot2*dt/2)
            xdot4, ydot4 = f(x + xdot3*dt, y + ydot3*dt)
            xnew = x + (xdot1 + 2*xdot2 + 2*xdot3 + xdot4)*dt/6
            ynew = y + (ydot1 + 2*ydot2 + 2*ydot3 + ydot4)*dt/6
            return xnew, ynew
        
        # Simulation Loop
        N = len(t)
        dt = t[1] - t[0]
        x = np.zeros(N)
        y = np.zeros(N)
        x[0] = x0
        y[0] = y0
        for k in range(N-1):
            x[k+1], y[k+1] = sim(x[k], y[k],dt)
            
            
        # add noise
        noise_x = np.random.standard_normal(len(x)) #normal distributed noise for X
        noise_y = np.random.standard_normal(len(x)) #normal distributed noise for Y
        
        time_series_data = np.empty([len(x),2])
        time_series_data[:,0] = x+noise_x*noise_factor_x + offset_x
        time_series_data[:,1] = y+noise_y*noise_factor_y + offset_y
        
    
        return time_series_data
    
    
    def fit_ODE(self, t, observations, x_init, y_init, lim_min, lim_max):
        """ fits Loktka Volterra model to time series data by solving an optimization problem
        e.g. --> runfile('DE.py')
        
        Input: 
                t: Time vector 
                observations: Observed time series data, containing x (Preys) data and y (Predators) data
                x_init: observed initial condition for x (Preys)
                y_init: observed initial condition for y (Predators)
                lim_min: minimum boundery condition for solving the optimization problem
                lim_max: maximum boundery condition for solving the optimization problem
    
        Output:
                fitted_mdl_result: fitted ODE model results
                residuum_ode: residuum of fitted ODE model 

        """

        #Random number generator for repeatabe results
        random.seed(0)
        
        def LV_model(parties, t, params):
            """ Definition of Lotka Volterra model, 
                described by four parameters, which are estimated by optimization
            """     
    

            #Estimated values
            a = params['alpha_est'].value
            b = params['beta_est'].value
            c = params['gamma_est'].value
            d = params['delta_est'].value
            
            #Domain formula for dx/dt & dy/dt
            result = [a*parties[0] - b*parties[0]*parties[1], 
                      c*parties[0]*parties[1] - d*parties[1]]
            
            return result
        
        
        def calculate_residuum(params, ts, observations):
            """ Time integration with calculation of residuum as optimization criterion
            """  
            
            #Time integration
            init = [params['x0'].value, params['y0'].value]
            ODE_mdl = scipy.integrate.odeint(LV_model, init, t, (params,)) 
            
            return ODE_mdl - observations
        
        # set parameters incl. optimization limits
        params = lmfit.Parameters()
        params.add('x0',         value= x_init, min=lim_min, max=lim_max)
        params.add('y0',         value= y_init, min=lim_min, max=lim_max)
        params.add('alpha_est',  value=1.0,     min=lim_min, max=lim_max)
        params.add('beta_est',   value=1.0,     min=lim_min, max=lim_max)
        params.add('gamma_est',  value=1.0,     min=lim_min, max=lim_max)
        params.add('delta_est',  value=1.0,     min=lim_min, max=lim_max)
        
        # fit model and find predicted values
        fitted_mdl = lmfit.minimize(calculate_residuum, params, args=(t, observations), method='leastsq')
        
        #Calculate output
        residuum_ode = fitted_mdl.residual.reshape(observations.shape)
        fitted_mdl_result = observations + residuum_ode
        
        
        return fitted_mdl_result, residuum_ode
    
    
    
    
    
    def visualize_ODE(self,t, alpha, beta, gamma, delta, nb_points):
        """ visualizes Loktka Volterra model with trajectory plot of X & Y
        e.g. --> runfile('DE.py')
        
        Input: 
                t: Time vector 
                alpha: estimated parameter for Lotka Volterra Model: positive, real constant
                beta:  estimated parameter  for Lotka Volterra Model: positive, real constant
                gamma: estimated parameter  for Lotka Volterra Model: positive, real constant
                delta: estimated parameter  for Lotka Volterra Model: positive, real constant
                nb_points: number of parallel arrows for visualization
        """
        #Thanks and reference to: https://github.com/scipy/scipy-cookbook/blob/master/ipython/LoktaVolterraTutorial.ipynb
        #https://github.com/scipy/scipy-cookbook/blob/master/LICENSE.txt 
        """
        Copyright (c) 2001, 2002 Enthought, Inc.
        All rights reserved.
        
        Copyright (c) 2003-2017 SciPy Developers.
        All rights reserved.
        
        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:
        
          a. Redistributions of source code must retain the above copyright notice,
             this list of conditions and the following disclaimer.
          b. Redistributions in binary form must reproduce the above copyright
             notice, this list of conditions and the following disclaimer in the
             documentation and/or other materials provided with the distribution.
          c. Neither the name of Enthought nor the names of the SciPy Developers
             may be used to endorse or promote products derived from this software
             without specific prior written permission.
        
        
        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
        ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
        BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
        OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
        SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
        INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
        CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
        ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
        THE POSSIBILITY OF SUCH DAMAGE.
        """


        #Random number generator for repeatabe results
        random.seed(0)
        
        #Visualization of ODE
        values  = np.linspace(0.3, 1.1, 5)                              # position of X0 between X_f0 and X_f1
        vcolors = plt.cm.autumn_r(np.linspace(0.3, 1., len(values)))    # colors for each trajectory
        
         
        def dX_dt(X, t=0):
            """ Return the growth rate of fox and rabbit populations. """
            return np.array([ alpha*X[0] -   beta*X[0]*X[1] ,
                          -gamma*X[1] + delta*beta*X[0]*X[1] ])
            
        X_f1 = np.array([ gamma/(delta*beta), alpha/beta])   
        #-------------------------------------------------------
        # plot trajectories
        for v, col in zip(values, vcolors):
            X0 = v * X_f1                               # starting point
            X = scipy.integrate.odeint( dX_dt, X0, t)         
            plt.plot( X[:,0], X[:,1], lw=3.5*v, color=col, label='X0,Y0=(%.2f, %.2f)' % ( X0[0], X0[1]) )
        
        #-------------------------------------------------------
        # define a grid and compute direction at each point
        ymax = plt.ylim(ymin=0)[1]                        # get axis limits
        xmax = plt.xlim(xmin=0)[1]
        
        
        xx = np.linspace(0, xmax, nb_points)
        yy= np.linspace(0, ymax, nb_points)
        
        X1 , Y1  = np.meshgrid(xx, yy)                  # create a grid
        DX1, DY1 = dX_dt([X1, Y1])                      # compute growth rate on the gridt
        M = (np.hypot(DX1, DY1))                        # Norm of the growth rate 
        M[ M == 0] = 1.                                 # Avoid zero division errors 
        DX1 /= M                                        # Normalize each arrows
        DY1 /= M
        
        #-------------------------------------------------------
        # Drow direction fields, using matplotlib 's quiver function
        plt.title('Coupled ODEs: trajectories and direction fields')
        Q = plt.quiver(X1, Y1, DX1, DY1, M, pivot='mid', cmap=plt.cm.jet)
        plt.xlabel('Preys', fontsize=16)
        plt.ylabel('Predators', fontsize=16)
        plt.legend()
        plt.grid()
        plt.xlim(0, xmax)
        plt.ylim(0, ymax)
        return 0
    
    
    
    
    def evaluate_prob_mdl(self, t,res_ode,ODE, observations, sigma_factor):
        """ fits a probabilistic model for uncertainty quantification
        e.g. --> runfile('DE.py')
        
        Input: 
                t: Time vector 
                res_ode: Residuum of ODE model
                ODE: ODE model which was fitted before 
                observations: observed labels, in our case the observations according to Lotka Volterra
                sigma_factor: definition of confidence limits of GP 
        """
        
        #Random number generator for repeatabe results
        random.seed(0)
        
        #Using tensorflow/probability
        prob = tfp.distributions
        psd_kernels = tfp.math.psd_kernels
        kernel = psd_kernels.ExponentiatedQuadratic()
        
        
        
        #Training on residuum of domain model as label    
        label_x = -res_ode
        std_est = np.std(label_x) #derived from Residuum
        
        
        #Probabilistic modelling with GP
        gprm = prob.GaussianProcessRegressionModel(kernel, t.reshape(-1,1), t.reshape(-1,1), np.array(label_x), observation_noise_variance=std_est )
        upper, lower = gprm.mean() + [sigma_factor * gprm.stddev(), -sigma_factor * gprm.stddev()]
        
        
        #Visualization
        plt.plot(t, label_x,'o',label='Noisy observations',alpha = 0.2)
        plt.plot(t, gprm.mean(),label='GP', linewidth=3)
        plt.fill_between(t, upper, lower, color='k', alpha=.1,label='Confidence')
        plt.legend(loc = 'upper right')
        plt.show()
        
        
        #probabilistic correction
        x_corr = gprm.mean() + ODE
        x_corr_up = upper + ODE
        x_corr_low = lower + ODE
        
        
        
        """
        #Visualization
        plt.plot(t, observation,'o',label='Noisy observations',alpha = 0.1)
        plt.plot(t,ODE ,label='ODE', linewidth=3)
        plt.plot(t,x_corr,'k-.',label='TF prob', linewidth=3)
        plt.fill_between(t, x_corr_up, x_corr_low, color='k', alpha=.1)
        plt.legend(loc = 'upper right')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.show()
        """
        
        
        #Residuum Distribution
        print("residuum: ODE - Label")
        print(np.mean(ODE- observations))
        print(np.std(ODE- observations))
        print("_______________")
        print("residuum: TF Prob - Label")
        print(np.mean(x_corr - observations))
        print(np.std(x_corr - observations))
        print("_______________")
        
        #visualization 
        self.visualize_results(t, x_corr, ODE, observations)
        return 0




    def optimize_mdl_prm(self, index_points,observation_index_points, observations, observation_noise_variance, sigma_factor):
        """ Optimize probabilistic model parameters via maximum marginal likelihood
        e.g. --> runfile('DE.py')
        
        Input: 
                index_points: Index points, in our case time series data: t.reshape(-1,1)
                observation_index_points: Index points, for which observations are available, in our case: t.reshape(-1,1) 
                observations: observed labels, in our case the observations according to Lotka Volterra
                observation_noise_variance: noise of observed labels
                sigma_factor: definition of confidence limits of GP 
                    
        Output: 
                gprm: optimized Gaussian Process (GP) model
                opt: mean of optimized GP
                upper: upper confidence limit of optimized GP
                lower: lower confidence limit of optimized GP
        """
        
        #Thanks and reference to: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/GaussianProcessRegressionModel?hl=fr
        
        
         #Random number generator for repeatabe results
        random.seed(0)

        #Tensorflow application
        tf.enable_v2_behavior()
        tfb = tfp.bijectors
        tfd = tfp.distributions
        psd_kernels = tfp.math.psd_kernels
        
        
        
        
        # Define a kernel with trainable parameters. Note we use TransformedVariable
        # to apply a positivity constraint.
        amplitude = tfp.util.TransformedVariable(
          1., tfb.Exp(), dtype=tf.float64, name='amplitude')
        length_scale = tfp.util.TransformedVariable(
          1., tfb.Exp(), dtype=tf.float64, name='length_scale')
        kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)
        
        observation_noise_variance = tfp.util.TransformedVariable(
            np.exp(-5), tfb.Exp(), name='observation_noise_variance')
        
        
        
        
        # Define a kernel with trainable parameters. Note we use TransformedVariable
        # to apply a positivity constraint.
        amplitude = tfp.util.TransformedVariable(
          1., tfb.Exp(), dtype=tf.float64, name='amplitude')
        length_scale = tfp.util.TransformedVariable(
          1., tfb.Exp(), dtype=tf.float64, name='length_scale')
        kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)
        
        observation_noise_variance = tfp.util.TransformedVariable(
            np.exp(-5), tfb.Exp(), name='observation_noise_variance')
        
        # We'll use an unconditioned GP to train the kernel parameters.
        gp = tfd.GaussianProcess(
            kernel=kernel,
            index_points=observation_index_points,
            observation_noise_variance=observation_noise_variance)
        
        optimizer = tf.optimizers.Adam(learning_rate=.05, beta_1=.5, beta_2=.99)
        
        #Optimization function
        @tf.function
        def optimize():
          with tf.GradientTape() as tape:
            loss = -gp.log_prob(observations)
          grads = tape.gradient(loss, gp.trainable_variables)
          optimizer.apply_gradients(zip(grads, gp.trainable_variables))
          return loss
        
        
        # We can construct the posterior at a new set of `index_points` using the same
        # kernel (with the same parameters, which we'll optimize below).   
        gprm = tfd.GaussianProcessRegressionModel(
            kernel=kernel,
            index_points=index_points,
            observation_index_points=observation_index_points,
            observations=observations,
            observation_noise_variance=observation_noise_variance)
        
        # First train the model, then draw and plot posterior samples.
        for i in range(90):
          neg_log_likelihood_ = optimize()
          if i % 5 == 0:
            print("Step {}: NLL = {}".format(i, neg_log_likelihood_))
        
        print("Final NLL = {}".format(neg_log_likelihood_))
        
        # ==> e.g.3 independently drawn, joint samples at `index_points`.
        nr_samples = 3
        samples = gprm.sample(nr_samples).numpy()
        
        """
        #Visualization
        plt.scatter(np.squeeze(observation_index_points), observations,label = 'Observations', alpha=.2)
        plt.plot(np.stack([index_points[:, 0]]*nr_samples).T, samples.T, c='r', alpha=.2,label = 'samples')
        plt.legend(loc = 'upper right')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.show()
        """

        
        #Output
        opt = gprm.mean()
        upper, lower =  opt + [sigma_factor * gprm.stddev(), -sigma_factor * gprm.stddev()]
        return gprm, opt, upper, lower







    def visualize_results(self, t, mdl, ODE, observations):
        """ Visualization of residuum plot and time series plot (each w & w/o probabilistic correction)
        e.g. --> runfile('DE.py')
        
        Input: 
                t: Time vector 
                mdl: optimized model, e.g. ODE w/ tuned probabilistic model
                ODE: ODE model as benchmark
                observations: observed labels, in our case the observations according to Lotka Volterra
        """
        
        
        sns.distplot(mdl - observations,kde=True,label = 'with correction')
        sns.distplot(ODE - observations,kde=True,label = 'ODE only')
        plt.xlabel('Model prediction -  Label', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.legend(fontsize=16)
        plt.title('Residuum Distribution', fontsize=16)
        plt.show()
        
        
        plt.plot(t, observations, 'o',label='Noisy observations',alpha = 0.2)
        plt.plot(t, ODE, '-', linewidth=7 ,label='ODE' ,alpha = 0.5)
        plt.plot(t, mdl, '-.', linewidth=4 ,label='Opt. Model Fit')
        plt.xlabel('Time', fontsize=16)
        plt.ylabel('Population', fontsize=16)
        plt.legend(loc = 'upper right')
        plt.show()
        
        return 0

#-----------------------------------------------------------------------------------------------------------------------------------
