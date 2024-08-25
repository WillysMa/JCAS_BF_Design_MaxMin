# -*- coding: utf-8 -*-
'''
# @File    : Converg_Fig.py
# @Author  : Mengyuan Ma
# @Contact: mamengyuan410@gmail.com
# @Time    : 2024/8/17 11:30
'''

import time

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from SysParams import *
from FuncRps import *
from Alg_Numpy import *
torch.manual_seed(666)
Kmax = 4
NumUE = Kmax // 2
n_iter_outer2 = n_iter_outer
n_iter_outer4 = n_iter_outer
if First_train:
     GenChannel(test_size, Nr, Nt, Kmax, M, C, data_path=data_path_test)

myModel = UnfoldNet(smoothing_factor, step_size, Nr, Nt, K, M, C, NoisePower, delta)

checkpoint = torch.load(model_file_name)
myModel.load_state_dict(checkpoint['model_state_dict'])



dataset = CustomDataset(file_path=data_path_test)
dataloader = DataLoader(dataset, batch_size=test_size)
myModel.eval()  # testing mode

n_iter_outer2 = 150
SNR_avr2 = np.zeros([6, 2])
obj_unfolding_all2 =np.zeros([test_size, 6, n_iter_outer2])
TC_unfolding_all2 = np.zeros((6,))

n_iter_outer4 = 150
SNR_avr4 = np.zeros([NumUE, 2])
obj_unfolding_all4 =np.zeros([test_size, NumUE, n_iter_outer4])
TC_unfolding_all4 = np.zeros((NumUE,))

obj_spawc_all_fs = np.empty([test_size,NumUE],dtype=object)
SNR_spawc_all_fs = np.zeros((test_size, NumUE, 2))

obj_spawc_all_ds = np.empty([test_size, NumUE],dtype=object)
SNR_spawc_all_ds = np.zeros((test_size, NumUE, 2))

obj_spawc_all_mm = np.empty([test_size, NumUE],dtype=object)
SNR_spawc_all_mm = np.zeros((test_size, NumUE, 2))

TC_algo_fs_all = np.zeros((test_size, NumUE))
TC_algo_ds_all = np.zeros((test_size, NumUE))
TC_algo_mm_all = np.zeros((test_size, NumUE))

for i_batch, data in enumerate(dataloader):
    H_batch, G_batch = data['Hall'], data['Gall']

    if SaveChMat:
        test_channel = {
            'Hall': H_batch,
            'Gall': G_batch
        }

        # Save the data to a .mat file
        scipy.io.savemat(directory_model+'test_data.mat', test_channel)

    for K in range(2, Kmax + 1,2):
        k_id = K//2-1
        Gs_batch = torch.zeros((test_size, Nr, Nt, M), dtype=torch.complex128)
        W_batch = torch.zeros((test_size, Nt, K), dtype=torch.complex128)
        F_batch = torch.zeros((test_size, Nr, M), dtype=torch.complex128)

        for ii in range(test_size):
            print(f'UE num:{K}, channel sample id:{ii}',flush=True)
            Hii, Gii = H_batch[ii, :, :K], G_batch[ii, :, :]

            Gs_ii, W_ii, F_ii = Initialize_WnF(Hii, Gii, Pt, Nr, Nt, M, C, NoisePower)
            Gs_batch[ii, :, :, :] = Gs_ii
            W_batch[ii, :, :] = W_ii
            F_batch[ii, :, :] = F_ii

            t0 = time.time()
            obj_iter_fs, snr_iter_fs, rate_iter_fs, rate_sum_fs = Alg_PGD(Hii.numpy(), Gii.numpy(), F_ii.numpy(), W_ii.numpy(), delta, Pt.numpy(), smoothing_factor.detach().numpy(), NoisePower, Iter=200,
                                                                          tolerance=1e-4, pgd_lr=step_size_fixed, BTS=False)


            t1 = time.time()
            tc_alg_fs = t1 - t0
            TC_algo_fs_all[ii, k_id] = tc_alg_fs
            obj_spawc_all_fs[ii, k_id] = obj_iter_fs
            SNR_spawc_all_fs[ii, k_id, :] = snr_iter_fs

            t0 = time.time()
            obj_iter_ds, snr_iter_ds, rate_iter_ds, rate_sum_ds = Alg_PGD(Hii.numpy(), Gii.numpy(), F_ii.numpy(), W_ii.numpy(), delta, Pt.numpy(), smoothing_factor.detach().numpy(), NoisePower, Iter=200,
                                                                          tolerance=1e-4, pgd_lr=step_size_fixed, BTS=True)
            t1 = time.time()
            tc_alg_ds = t1 - t0
            TC_algo_ds_all[ii, k_id] = tc_alg_ds
            # obj_spawc_all.append(obj_iter)
            obj_spawc_all_ds[ii, k_id] = obj_iter_ds
            SNR_spawc_all_ds[ii, k_id, :] = snr_iter_ds

            t0 = time.time()
            obj_iter_mm, snr_iter_mm, rate_iter_mm, rate_sum_mm = Alg_Heuristic(Hii.numpy(), Gii.numpy(), F_ii.numpy(), W_ii.numpy(), delta, Pt.numpy(), smoothing_factor.detach().numpy(), NoisePower, Iter=200, tolerance=1e-4)
            t1 = time.time()
            tc_alg_mm = t1 - t0
            TC_algo_mm_all[ii, k_id] = tc_alg_mm
            # obj_spawc_all.append(obj_iter)
            obj_spawc_all_mm[ii, k_id] = obj_iter_mm
            SNR_spawc_all_mm[ii, k_id, :] = snr_iter_mm

        t0 = time.time()
        loss_list, Obj_cache, Rate_cache, SNR_cache = myModel(H_batch[:, :, :K], G_batch, Gs_batch, F_batch, W_batch, Pt, NoisePower, delta, n_iter_W=2,NumIter_outer=n_iter_outer2)
        t1 = time.time()
        tc_unfolding = t1 - t0
        TC_unfolding_all2[k_id] = tc_unfolding
        obj_unfolding_all2[:, k_id, :] = Obj_cache
        SNR_out = pow2db(np.mean(SNR_cache[:, -1, :].squeeze(), axis=0))
        SNR_avr2[k_id, :] = SNR_out

        t0 = time.time()
        loss_list, Obj_cache, Rate_cache, SNR_cache = myModel(H_batch[:, :, :K], G_batch, Gs_batch, F_batch, W_batch, Pt, NoisePower, delta, n_iter_W=4, NumIter_outer=n_iter_outer4)
        t1 = time.time()
        tc_unfolding = t1 - t0
        TC_unfolding_all4[k_id] = tc_unfolding
        obj_unfolding_all4[:, k_id, :] = Obj_cache
        SNR_out = pow2db(np.mean(SNR_cache[:, -1, :].squeeze(), axis=0))
        SNR_avr4[k_id, :] = SNR_out

        pass

    break #the first test_size channel
K_id = 1  # UE (K_id+1)*2
# channel_id = 1

obj_unfolding_avr2 = np.mean(obj_unfolding_all2[:, K_id, :], axis=0)
obj_unfolding_avr4 = np.mean(obj_unfolding_all4[:, K_id, :], axis=0)

obj_alg_fs_all = obj_spawc_all_fs[:, K_id]
obj_alg_ds_all = obj_spawc_all_ds[:, K_id]
obj_alg_mm_all = obj_spawc_all_mm[:, K_id]

ccc = 1
obj_alg_fs_avr = average_padded_lists_1d(obj_alg_fs_all)
obj_alg_ds_avr = average_padded_lists_1d(obj_alg_ds_all)
obj_alg_mm_avr = average_padded_lists_1d(obj_alg_mm_all)

plt.figure()
xx = list(range(len(obj_unfolding_avr2)))
plt.plot(xx, obj_unfolding_avr2, 'g-*', label='Unfolding (I=2)')
plt.plot(xx, obj_unfolding_avr4, 'r-*', label='Unfolding (I=4)')

xx = list(range(len(obj_alg_fs_avr)))
plt.plot(xx, obj_alg_fs_avr, 'b-s', label='PGD (fixed step size)')

xx = list(range(len(obj_alg_ds_avr)))
plt.plot(xx, obj_alg_ds_avr, 'k-o', label='PGD (dynamic step size)')

xx = list(range(len(obj_alg_mm_avr)))
plt.plot(xx, obj_alg_mm_avr, 'm-x', label='Heuristic')

plt.xlabel('Number of iterations')
plt.ylabel('Objective value')
plt.legend(prop={'size': 10, 'weight': 'bold'})
plt.grid(True)
plt.box(True)
fig_name = 'Convergence'+'_K'+str((K_id+1)*2)+ '_IterW'+str(n_iter_W)+'_IterOuter'+str(n_iter_outer)
plt.savefig(directory_model + fig_name +'.png')  # save figure
plt.savefig(directory_model + fig_name +'.eps')  # save figure
plt.show()

data = {
    'Converg_unfolding2': obj_unfolding_avr2,
    'Converg_unfolding4': obj_unfolding_avr4,
    'Converg_Alg_fs': obj_alg_fs_avr,
    'Converg_Alg_ds': obj_alg_ds_avr,
    'Converg_Alg_mm': obj_alg_mm_avr,
    'delta': delta,
    'n_iter_W': n_iter_W,
    'n_iter_outer': n_iter_outer
    }

# Save the data to a .mat file
file_name = 'Converge_delta'+str(delta)+'.mat'
scipy.io.savemat(directory_model+file_name, data)
pass

print('-----------------------finished------------------------',flush=True)