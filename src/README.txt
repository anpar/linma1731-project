This directory contains all the source code used to produce the figures of the report.

The code is written in Python 3 and uses the following scientific libraries: NumPy, Scipy
Matplotlib.

To run a script: simply open a terminal, cd to this directory and type in
$ python3 <script name>
where script name can be anything in
- q2_3d_trajectory.py : to generate the 3D trajectory asked in question 2;
- q2_mes_vs_real.py : to simulate the system and its observation during 50s;
- q3_smc.py : to run the SMC algorithm during 16s with 50, 100 and 1000 particles. This
will outputs histograms, error curves, particles in a 3D plot and trajectory of the first
coordinate (real, measured and estimated).
- q4_exp.py : to generate the plot of the part "experiment" of the report.
Parameters in this script are really easy to change if you want to experiment more.
- q5_smc_data.py : simply apply our SMC algorithm on the provided data.
- q6_ekf : to generate estimated trajectories with EKF + error curves.
- q7_dist.py : to generate the plots used in the comparison of the distribution.
