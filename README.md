# ODE_Analysis
Simulate, solve and visualize nonlinear differential equations utilizing probabilistic models that are optimized via maximum marginal likelihood.        

 
# Usage of RUN_prob-ode:
* The class ODE_Analysis enables the simulation of coupled ODEs with additional observation uncertainties. 
The goal is to describe Lotka Volterra sequences with an ODE-solver and a probabilistic model. 
In addition, we want to analyze and visualize results.

![alt text](https://github.com/christiansimonis/prob-ode/blob/master/vis/direction_field.JPG)


* A domain-driven time integration approach is applied to model our coupled set of ODEs. The ODE-solver is combined with a probabilistic model. The residuum of the pure ODE-solver and probabilistically boosted model is depicted here:

![alt text](https://github.com/christiansimonis/prob-ode/blob/master/vis/residuum.JPG)


* The parameters of our probabilistically boosted model are optimized by tuning the maximum marginal likelihood. Time series behavior of the optimized system model is plotted with associated noisy observations:

![alt text](https://github.com/christiansimonis/prob-ode/blob/master/vis/time_series.JPG)

* A trajectory plot with samples from the optimized system model is shown:

![alt text](https://github.com/christiansimonis/prob-ode/blob/master/vis/trajectory.JPG)



        

# Disclaimer
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Acknowledgements 
* https://www.tensorflow.org/probability
* https://github.com/scipy/scipy-cookbook
* https://github.com/smkalami
* https://en.wikipedia.org/wiki/Marginal_likelihood



Thanks and reference to:
(Name,                                 Version,                      License)  
* numpy,                               1.19.5,                       BSD,                                         Copyright (c) 2005-2020, NumPy Developers
* tensorflow-probability,              0.12.2,                       Apache Software License , Version 2.0,       Copyright 2018 The TensorFlow Probability Authors.  All rights reserved
* tensorflow,                          2.5.0,                        Apache Software License,                     Copyright 2019 The TensorFlow Authors.  All rights reserved.                                                                                                                                                                                                                                                                                                                                                                                                                                                        
* matplotlib,                          3.4.2,                        Python Software Foundation License,          Copyright (c) 2002 - 2012 John Hunter, Darren Dale, Eric Firing, Michael Droettboom and the Matplotlib development team; 2012 - 2021 The Matplotlib development team.
* seaborn,                             0.11.1,                       BSD 3-Clause "New" or "Revised" License,     Copyright (c) 2012-2021, Michael L. Waskom.
* scipy,                               1.5.2,                        BSD 3-Clause "New" or "Revised" License,     Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.                         
* lmfit/lmfit-py,                      1.0.2,                        BSD-3                                        Copyright 2021 Matthew Newville, The University of Chicag, Renee Otten, Brandeis University, Till Stensitzki, Freie Universitat Berlin, A. R. J. Nelson, Australian Nuclear Science and Technology Organisation, Antonino Ingargiola, University of California, Los Angeles, Daniel B. Allen, Johns Hopkins University, Michal Rawlik, Eidgenossische Technische Hochschule, Zurich.
* app (Lotka Volterra simulation),       -,                          MIT License,                                 Copyright (c) 2018 Mostapha Kalami Heris / Yarpiz Team.
* scipy/scipy-cookbook,                  -,                          Copyright (c) 2001, 2002 Enthought, Inc.,    All rights reserved., Copyright (c) 2003-2017 SciPy Developers.

# Contact and collaboration
* [LinkedIn](https://www.linkedin.com/in/christiansimonis/)
* [GitHub](https://github.com/login?return_to=%2Fchristiansimonis)
* [Forks](https://github.com/christiansimonis/prob-ode/network/members)
* [Stargazers](https://github.com/christiansimonis/prob-ode/stargazers)
