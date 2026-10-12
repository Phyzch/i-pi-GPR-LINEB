# Gaussian Process Regression Enhanced Line Integral Nudged Elastic Band method code.
The software is built on i-PI.

See README.md docs in /i-pi/ for setup information.

## 1. Publications:
(1) Accelerating Instanton Theory with the Line Integral Nudged Elastic Band Method and Gaussian Process Regression (J. Chem. Theory Comput. (2025) 21 (15): 7517–7534.)

(2) A Machine Learning Framework for Efficient Ring-Polymer Instanton Rate Calculations (
https://doi.org/10.48550/arXiv.2602.16962)

## 2. FEATURE: 
### 2.1  Line Integral Nudged Elastic Band

We have implemented Line Integral Nudged Elastic Band method in i-PI software, the code can be found in:
- ./i-pi/ipi/engine/motion/neb_instanton.py and neb_instanton_gpr.py 

- ./i-pi/ipi/inputs/motion/neb_instanton.py and neb_instanton_gpr.py 

### 2.2 Gaussian Process Regression model 

We have implemented Gaussian Process Regression to accelerate the Line Integral Nudged Elastic Band algorithm. the code can be found in:
./i-pi/gpy/

Here, there are two GPR models available:


(1) The GPR model predict potential and forces:

- ./i-pi/gpr/gprtools.py 

-  ./i-pi/gpr/gprGrad

(2) The GPR model predict potentials, forces and Hessians:

- ./i-pi/gpr/gpr_hessian_tools.py

- ./i-pi/gpr/gprHessian/











## 2.3 Bayesian Optimization

We perform the Bayesian Optimization combining GPR model and LI-NEB. 

The GPR model generate the surrogate PES, on which the LI-NEB algorithm search the instanton path. The code that performs the Bayesian Optimization is located in:

- ./i-pi/gpr_script/

Here **neb_gpr.py** use the scripting function in i-pi to perform the active learning, **gpr_interface.py** is the interface that connecting LI-NEB path searching algorithm and the two GPR models we use.

## 3. Usage ##
(1) Source the env.sh file

`source ./i-pi/env.sh`

(2) Run the active learning code. The entry file for scripting is `./i-pi/gpr_script/neb_gpr.py`

`python3 -u ./i-pi/gpr_script/neb_gpr.py input.xml` 

(3) Run the electronic structure code that handshake with the i-PI. 


