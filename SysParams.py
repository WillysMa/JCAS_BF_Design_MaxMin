# -* coding: utf-8 -*-
'''
@File；SysParams.py
@Author: Mengyuan Ma
@Contact: mamengyuan410@gmail.com
@Time: 2022-08-15 15:45
Configure system parameters
'''
import os
import torch
import datetime
import inspect
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                Parameters
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# ///////////////////////System parameters //////////////////////////////
Nt = 16
Nr = 16
K = 4  # users
M = 2  # sensing targets
C = 2  # clutters
First_train = False  # set it to True to generate data before first training
SaveChMat = False
NoisePower = 1.  # normalized noise power
SNR_dB = 20  # [0., 2., 4., 6., 8., 10., 12.]
Pt = torch.tensor(10 ** (SNR_dB / 10)) # normalized transmit power
delta = 1

# Training parameters
Seed_train = 1
Seed_test = 101

# ////////////////////////////////////////////// MODEL PARAMS //////////////////////////////////////////////
SUM_LOSS = 1

train_size = 500    # size of training set
test_size = 50      # size of testing set
batch_size_train = 32   # batch size when training
n_epoch = 30      # number of training epochs
n_iter_iter = 3   # number of layer for obtain W
n_iter_outer = 150   # number of layer for obtain W
n_iter_W = 2

start_learning_rate = 3e-4
Log_interval = 10  # For every Log_interval batches, print loss
Weight_decay = 1e-2 # add L2 regularizer to weight, the penalty is larger with high Weight_decay
Lr_decay_rate = 0.97  # decrease the learning rate exponentially

# ========================== initiate step sizes as tensor for training ================
step_size_fixed = 0.01  # step size of conventional PGA
step_size = torch.full([n_iter_outer, n_iter_iter], step_size_fixed, requires_grad=True)
mu = 10.
smoothing_factor = torch.tensor([mu, mu], requires_grad=True)


# ////////////////////////////////////////////// SAVING RESULTS AND DATA //////////////////////////////////////////////
time_signature = datetime.datetime.now().strftime('%Y%m%d_%H%M')
system_config = 'Nt'+str(Nt)+'Nr'+str(Nr)+'K'+str(K)+'M'+str(M)+'C'+str(C)
train_data_file_name = "train_data.npz"
test_data_file_name = "test_data.npz"

directory_data = "./"+ system_config + "/"
if not os.path.exists(directory_data):
    os.makedirs(directory_data)

data_path_train = directory_data + train_data_file_name
data_path_test = directory_data + test_data_file_name

# To save trained model
directory_model = "./" + system_config + "/delta"+str(delta) + "/"
if not os.path.exists(directory_model):
    os.makedirs(directory_model)

model_file_name = directory_model + 'trained_model.pth'


log_file_path = directory_model + 'Logs_Info.txt'
def save_current_variables(dir_para_file=log_file_path):
    # Get the current frame's global variables
    current_vars = inspect.currentframe().f_back.f_globals

    with open(dir_para_file, 'w') as f:
        for name, value in current_vars.items():
            if not name.startswith('__'):
                f.write(f'{name} = {value}\n')

def add_logs(infos, dir_para_file=log_file_path):
    if isinstance(infos, str):
        file_para = open(dir_para_file, 'a')
        file_para.write('\n')
        file_para.write(infos + '\n')
        file_para.close()
    else:
        raise TypeError(f"Invalid input type: {type(infos).__name__}. Expected a string.")
    pass

# Call the function
