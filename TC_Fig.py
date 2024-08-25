# -*- coding: utf-8 -*-
'''
# @File    : TC_Fig.py
# @Author  : Mengyuan Ma
# @Contact: mamengyuan410@gmail.com
# @Time    : 2024/8/17 12:45
'''
# -*- coding: utf-8 -*-

import time

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from SysParams import *
from FuncRps import *
from Alg_Numpy import *
torch.manual_seed(666)
Kmax = 12
NumUE = Kmax // 2
n_iter_outer2 = n_iter_outer
n_iter_outer4 = 100

if First_train:
     GenChannel(test_size, Nr, Nt, Kmax, M, C, data_path=data_path_test)

myModel = UnfoldNet(smoothing_factor, step_size, Nr, Nt, K, M, C, NoisePower, delta)

checkpoint = torch.load(model_file_name)
myModel.load_state_dict(checkpoint['model_state_dict'])

dataset = CustomDataset(file_path=data_path_test)
dataloader = DataLoader(dataset, batch_size=test_size)
myModel.eval()  # testing mode


SNR_avr2 = np.zeros([NumUE, 2])
obj_unfolding_all2 =np.zeros([test_size, NumUE, n_iter_outer2])
TC_unfolding_all2 = np.zeros((NumUE,))


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

    break # channel num is test_size
TC_algo_fs_avr = np.mean(TC_algo_fs_all, axis=0)
TC_algo_ds_avr = np.mean(TC_algo_ds_all, axis=0)
TC_algo_mm_avr = np.mean(TC_algo_mm_all, axis=0)
TC_unfolding_avr2 = TC_unfolding_all2 / test_size
TC_unfolding_avr4 = TC_unfolding_all4 / test_size

xx = np.array(list(range(2, Kmax + 1, 2)))
width = 0.3  # width of the bars
plt.figure()
plt.bar(xx, TC_algo_fs_avr, width=width, color='b', label='PGD (fixed step size)')
plt.bar(xx + width, TC_algo_ds_avr, width=width, color='g', label='PGD (dynamic step size)')
plt.bar(xx + 2 * width, TC_algo_mm_avr, width=width, color='m', label='Heuristic')
plt.bar(xx + 3 * width, TC_unfolding_avr2, width=width, color='r', label='Unfolding (I=2)')
plt.bar(xx + 4 * width, TC_unfolding_avr4, width=width, color='k', label='Unfolding (I=4)')

plt.xlabel('Number of communication users')
plt.ylabel('Average time cost [s]')
plt.xticks(xx + width, xx)  # Set the x-ticks to be centered
plt.legend(prop={'size': 10, 'weight': 'bold'},)
# plt.grid(True)
plt.box(True)
fig_name = 'TCvsUE'
plt.savefig(directory_model + fig_name+'.png')  # save figure
plt.savefig(directory_model + fig_name+'.eps')
plt.show()

SNR_alg_fs = np.mean(SNR_spawc_all_fs, axis=0)
SNR_alg_ds = np.mean(SNR_spawc_all_ds, axis=0)
SNR_alg_mm = np.mean(SNR_spawc_all_mm, axis=0)
xx = list(range(2, Kmax + 1,2))
plt.figure()
h1, = plt.plot(xx, SNR_avr2[:, 0], 'b-s',  markersize=5)
h2, = plt.plot(xx, SNR_avr4[:, 0], 'gp',  markersize=5)
h3, = plt.plot(xx, SNR_alg_fs[:, 0], 'k-x', markersize=5)
h4, = plt.plot(xx, SNR_alg_ds[:, 0], 'c-^',  markersize=5)
h5, = plt.plot(xx, SNR_alg_mm[:, 0], 'm-+', markersize=5)

h11, = plt.plot(xx, SNR_avr2[:, 1], 'b:o',  markersize=5)
h22, = plt.plot(xx, SNR_avr4[:, 1], 'rh',  markersize=5)
h33, = plt.plot(xx, SNR_alg_fs[:, 1], 'k:d', markersize=5)
h44, = plt.plot(xx, SNR_alg_ds[:, 1], 'c:*',  markersize=5)
h55, = plt.plot(xx, SNR_alg_mm[:, 1], 'm:+', markersize=5)
plt.xlabel('Number of communication users')
plt.ylabel('min SINR or min SCNR [dB]')
plt.legend(prop={'size': 10, 'weight': 'bold'})

l1 = plt.legend([h1, h2, h3, h4, h5], ['SINR (Unfolding I=2)', 'SINR(Unfolding I=4)', 'SINR (PGD with fixed step size)',
                                       'SINR (PGD with dynamic step size)', 'SINR (Heuristic)'],
           prop={'size': 10, 'weight': 'bold'}, loc='lower left', bbox_to_anchor=(0.0, 0.0))
plt.gca().add_artist(l1)
# Second legend

plt.legend([h11, h22, h33, h44, h55], ['SCNR (Unfolding I=2)', 'SCNR(Unfolding I=4)', 'SCNR (PGD with fixed step size)',
                                       'SCNR (PGD with dynamic step size)', 'SCNR (Heuristic)'],
           prop={'size': 10, 'weight': 'bold'}, loc='upper right', bbox_to_anchor=(1.0, 1.0))
plt.grid(True)
plt.box(True)
fig_name = 'SNRvsUE'
plt.savefig(directory_model + fig_name+'.png')  # save figure
plt.savefig(directory_model + fig_name+'.eps')  # save figure
plt.show()


data = {
    'TC_Alg_fs': TC_algo_fs_avr,
    'TC_Alg_ds': TC_algo_ds_avr,
    'TC_Alg_mm': TC_algo_mm_avr,
    'TC_Unfolding2': TC_unfolding_avr2,
    'TC_Unfolding4': TC_unfolding_avr4,
    'SNR_Alg_fs': SNR_alg_fs,
    'SNR_Alg_ds': SNR_alg_ds,
    'SNR_Alg_mm': SNR_alg_mm,
    'SNR_Unfolding2': SNR_avr2,
    'SNR_Unfolding4': SNR_avr4,
    'delta': delta,
    'n_iter_W': n_iter_W,
    'n_iter_outer': n_iter_outer
    }

# Save the data to a .mat file
file_name = 'TCnSNR'+str(delta)+'.mat'
scipy.io.savemat(directory_model+file_name, data)
pass

print('-----------------------finished------------------------',flush=True)