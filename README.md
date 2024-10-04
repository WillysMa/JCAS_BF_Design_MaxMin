# Simulator for "Model-based Machine Learning for Max-Min Fairness Beamforming Design in JCAS Systems"

### Important information

We use python 3.12 and Pytorch 2.4 and for simulations.

If you find simulator (or parts of it) helpful for your publication, please kindly cite our paper:

M. Ma, T. Fang, N. Shlezinger, A. Swindlehurst, M. Juntti, and N. Nguyen, “Model-based machine learning for max-min
fairness beamforming design in JCAS systems,” arXiv preprint arXiv:2409.17644, 2024.

# How to start a simulation
Download the whole project to your local worksation.

Run the 'TC_Fig.py' and 'Converg_Fig.py'. You will get the figures of convergence, time cost, and SINR(SCNR) comparison.




# How to retrain a new model?
- Config 'SysParams.py'
- Run the 'main.py'
You will get the data that can be compared to other schemes. You can specify your own system and simulation parameters. More details about the parameters are suggestted to read the paper. 

We highly recommend you to execute the code step-by-step (using PyCharm's debug mode) in order to get a detailed understanding of the simulator.

### Version history
- Version 0.1: mengyuan.ma@oulu.fi - simplified/commented code for GitHub
