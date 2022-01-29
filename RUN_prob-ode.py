""" 2)  RUN_prob-ode: The purpose of this code to demonstrate, how to simulate, 
        solve and visualize differential equations, utilizing probabilistic models."""
#-----------------------------------------------------------------------------------------------------------------------------------
__author__ = "Christian Simonis"
__copyright__ = "Copyright 2022"
__version__ = "1.1"
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
import random                                                      # https://docs.python.org/3/library/random.html                                            
import json                                                        # https://docs.python.org/3/library/json.html

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

# Defined Lotka-Volterra Model parameters for creating time series data
alpha = 1.0
beta = 0.55
gamma = 1.15
delta = 0.75


# Simulation Time
N = 800
till = 15
t = np.linspace(0,till,N)
dt = t[1] - t[0]

# Initial Values
x0 = 1.3
y0 = 0.5

#Observability, assuming a noisy signal
noise_factor_x = 0.2
noise_factor_y = 0.3
offset_x = 0.9 #Preys that do not take part in the game
offset_y = 1.2 #Predators that do not  not take part in the game

#Probabilistic analysis
sigma_eval = 3 #Evaluation at 3 sigma


#Visualization of trajectory samples
nr_samples = 10

#-----------------------------------------------------------------------------------------------------------------------------------
#import ODE_Analysis
from ODE_Analysis import ODE_Analysis
#-----------------------------------------------------------------------------------------------------------------------------------

#Random number generator for repeatabe results
random.seed(0)



#call class ODE_Analysis
Lotka_Volterra = ODE_Analysis()


#visualization of pure theoretic ODE part based on parameters 
nb_points   = 20
Lotka_Volterra.visualize_ODE(t, alpha, beta, gamma, delta, nb_points)
plt.show()


#create data with noisy observations and apply term which ODE model cannot describe (offset)
data = Lotka_Volterra.create_time_series(alpha, beta, gamma, delta, t, x0, y0, noise_factor_x, noise_factor_y, offset_x, offset_y)
x_init = data[0,0]
y_init = data[0,1]


#optimization: Fit differential eqaution model
lim_min = 0
lim_max = 20
fitted_mdl_result, residuum_ode = Lotka_Volterra.fit_ODE(t, data, x_init, y_init, lim_min, lim_max)
x_res_ode = residuum_ode[:,0]
y_res_ode = residuum_ode[:,1]





# plot data and fitted curves
plt.plot(t, data, 'o',label='Noisy observations',alpha = 0.2)
plt.plot(t, fitted_mdl_result, '-.', linewidth=5 ,label='Model Fit')
plt.xlabel('Time', fontsize=16)
plt.ylabel('Population', fontsize=16)
plt.legend(loc = 'upper right')
plt.show()




# for X (Preys)
ODE_x = fitted_mdl_result[:,0]
observation_x = data[:,0]
Lotka_Volterra.evaluate_prob_mdl(t,x_res_ode,ODE_x, observation_x, sigma_eval)


# for Y (Predators)
ODE_y = fitted_mdl_result[:,1]
observation_y = data[:,1]
Lotka_Volterra.evaluate_prob_mdl(t,y_res_ode,ODE_y, observation_y, sigma_eval)




#Optimization input
observation_index_points = t.reshape(-1,1) 
index_points = t.reshape(-1,1)


#Call function for X optimization
print("--------------Optimization for Prey class X--------------")
obs_X = -x_res_ode
observation_noise_variance_X = np.std(obs_X)
GP_opt_X, x_opt_mean, x_opt_up, x_opt_low = Lotka_Volterra.optimize_mdl_prm(index_points,observation_index_points, obs_X, observation_noise_variance_X, sigma_eval)


#probabilistic correction for X 
x_opt = x_opt_mean + ODE_x
x_opt_up = x_opt_up + ODE_x
x_opt_low = x_opt_low + ODE_x


#Visualization    
Lotka_Volterra.visualize_results(t, x_opt, ODE_x, observation_x)


#Call function for Y optimization
print("-----------Optimization for Predator class Y-------------")
obs_Y = -y_res_ode
observation_noise_variance_Y = np.std(obs_Y)
GP_opt_Y, y_opt_mean, y_opt_up, y_opt_low = Lotka_Volterra.optimize_mdl_prm(index_points,observation_index_points, obs_Y, observation_noise_variance_Y, sigma_eval)



#probabilistic correction for Y
y_opt = y_opt_mean + ODE_y
y_opt_up = y_opt_up + ODE_y
y_opt_low = y_opt_low + ODE_y

#Visualization
Lotka_Volterra.visualize_results(t, y_opt, ODE_y, observation_y)
print("-----------------Optimization finished-------------------")





#Time series plot
plt.plot(t, data, 'o',label='Noisy observations',alpha = 0.2)
plt.plot(t, x_opt, c='r', alpha=1, linewidth=3, label = 'Model for X (Prey class)')
plt.fill_between(t, x_opt_up, x_opt_low, color='r', alpha=.1)
plt.plot(t, y_opt, c='k', alpha=1, linewidth=3, label = 'Model for Y (Predator class)')
plt.fill_between(t, y_opt_up, y_opt_low, color='k', alpha=.1)
plt.legend(ncol=2)
plt.xlabel('Time', fontsize=16)
plt.ylim(0,8)
plt.ylabel('Population', fontsize=16)
plt.title('Optimized probabilistic model, combined with ODE model')
plt.show()


#Sampling from probabilistic model
samples_X = GP_opt_X.sample(nr_samples).numpy()
samples_Y = GP_opt_Y.sample(nr_samples).numpy()


#Diagram with trajectories    
plt.plot(x_opt, y_opt, c='b', alpha=.5, linewidth=7, label = 'Model Mean') 
plt.plot(samples_X[0,:]  + ODE_x, samples_Y[0,:]  + ODE_y, c='b', alpha=.1, label = 'Model samples') #for creating one legend entry
plt.plot(samples_X.T + ODE_x.reshape(-1,1), samples_Y.T + ODE_y.reshape(-1,1), c='b', alpha=.1, label = '_nolegend_') # no legend entrys hier
plt.legend(loc = 'upper right')
plt.xlabel('Preys', fontsize=16)
plt.ylabel('Predators', fontsize=16)
plt.title('Trajectory diagram w/ {} drawn samples from probabilistic model'.format(nr_samples))
plt.show()



#JSON export
save_output = np.zeros((len(t),3))
save_output[:,0] = t        # Time 
save_output[:,1:] = data    # Lotka Volterra simulations

#savind file
json_str = json.dumps(save_output.tolist())
with open('data.json', 'w') as outfile:
    outfile.write(json_str)