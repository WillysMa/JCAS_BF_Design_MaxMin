# -*- coding: utf-8 -*-
'''
# @File    : Main.py
# @Author  : Mengyuan Ma
# @Contact: mamengyuan410@gmail.com
# @Time    : 2024/8/2 11:56
'''
import time

# import torch
from torch.utils.data import DataLoader
from SysParams import *
from FuncRps import *
from Alg_Numpy import *
from torch.optim.lr_scheduler import ExponentialLR

save_current_variables()

time_now_start = datetime.datetime.now().strftime('%Y-%m-%d %H-%M%S')
print('The trainings starts at the time:', time_now_start, flush=True)  # 当前时间
add_logs('The training starts at the time= ' + str(time_now_start))

t_start = time.time()
torch.manual_seed(666)

if First_train:
    GenChannel(train_size, Nr, Nt, K, M, C, data_path=data_path_train)

dataset = CustomDataset(file_path=data_path_train)
dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=True)

myModel = UnfoldNet(smoothing_factor, step_size, Nr, Nt, K, M, C, NoisePower, delta)


myModel.train()
optimizer = torch.optim.Adam(myModel.parameters(), lr=start_learning_rate, weight_decay=Weight_decay)

# Define scheduler
scheduler = ExponentialLR(optimizer, gamma=Lr_decay_rate)

batch_count = 0
Loss_cache = []
Lr_list = []
for i_epoch in range(n_epoch):
    # print(i_epoch)
    for i_batch, data in enumerate(dataloader):
        batch_count += 1
        H_batch, G_batch = data['Hall'], data['Gall']

        batch_size_train, _, _ = H_batch.shape
        Gs_batch = torch.zeros((batch_size_train, Nr, Nt, M), dtype=torch.complex128)
        W_batch = torch.zeros((batch_size_train, Nt, K), dtype=torch.complex128)
        F_batch = torch.zeros((batch_size_train, Nr, M), dtype=torch.complex128)

        for ii in range(batch_size_train):
            Hii, Gii = H_batch[ii, :, :], G_batch[ii, :, :]

            Gs_ii, W_ii, F_ii = Initialize_WnF(Hii, Gii, Pt, Nr, Nt, M, C, NoisePower)
            Gs_batch[ii, :, :, :] = Gs_ii
            W_batch[ii, :, :] = W_ii
            F_batch[ii, :, :] = F_ii

        # Forward pass: Compute predicted y by passing x to the model
        obj_list, _, _, _ = myModel(H_batch, G_batch, Gs_batch, F_batch, W_batch, Pt, NoisePower, delta, n_iter_W, n_iter_outer)

        if SUM_LOSS:
            loss = sum(obj_list)
        else:
            loss = obj_list[-1]

        Loss_cache.append(loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

        if batch_count % Log_interval == 0:              # print arverage loss
            len_loss = len(Loss_cache)
            if len_loss > 2 * Log_interval:
                avr_loss = np.mean(Loss_cache[len_loss - Log_interval:])  # 取倒数Log_interval个loss做平均
                print(
                    f'Epoch:{i_epoch}, batch_id:{i_batch}, learning rate: {Lr_list[-1]:.5f}, average loss:{avr_loss:.6f}',
                    flush=True)
    scheduler.step()
checkpoint = {
    'Epoch': i_epoch,
    'Batch': i_batch,
    'loss': Loss_cache,
    'model_state_dict': myModel.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()}
torch.save(checkpoint, model_file_name)  # save model

time_now_end = datetime.datetime.now().strftime('%Y-%m-%d %H-%M%S')
print('The trainings ends at the time:', time_now_end,flush=True)  # 当前时间
t_end = time.time()
time_cost = (t_end - t_start)/3600
print(f'---------End training------time cost: {time_cost:.4f}h',flush=True)
# infos = 'The training ends at the time= ' + str(time_now_end)
add_logs('The training ends at the time= ' + str(time_now_end))
add_logs('Training time cost =' + str(time_cost))

xx = list(range(len(Loss_cache)))
plt.figure()
plt.plot(xx, Loss_cache, 'r-', label='-obj')
plt.xlabel('Number of batches')
plt.ylabel('Loss')
plt.legend(prop={'size': 10, 'weight': 'bold'})
plt.grid(True)
plt.box(True)

fig_name = 'loss_batch.png'
fig_path = directory_model + fig_name
plt.savefig(fig_path)  # save figure
plt.show()
#////////////////////////////////test/////////////////////////////////
print('-----------------------test begin-------------------------',flush=True)

t_start = time.time()
Kmax = 4
NumUE = Kmax // 2
# n_iter_outer = 50
if First_train:
     GenChannel(test_size, Nr, Nt, Kmax, M, C, data_path=data_path_test)
dataset = CustomDataset(file_path=data_path_test)
dataloader = DataLoader(dataset, batch_size=test_size)
myModel.eval()  # testing mode

SNR_avr = np.zeros([NumUE, 2])
obj_unfolding_all =np.zeros([test_size, NumUE, n_iter_outer])

obj_spawc_all_fs = np.empty([test_size,NumUE],dtype=object)
SNR_spawc_all_fs = np.zeros((test_size, NumUE, 2))

obj_spawc_all_ds = np.empty([test_size, NumUE],dtype=object)
SNR_spawc_all_ds = np.zeros((test_size, NumUE, 2))

TC_algo_fs_all = np.zeros((test_size, NumUE))
TC_algo_ds_all = np.zeros((test_size, NumUE))
TC_unfolding_all = np.zeros((NumUE,))
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
            obj_iter_fs, snr_iter_fs, rate_iter_fs, _ = Alg_PGD(Hii.numpy(), Gii.numpy(), F_ii.numpy(), W_ii.numpy(), delta, Pt.numpy(), smoothing_factor.detach().numpy(), NoisePower, Iter=200,
                                                                tolerance=1e-4, pgd_lr=step_size_fixed, BTS=False)
            t1 = time.time()
            tc_alg_fs = t1 - t0
            TC_algo_fs_all[ii, k_id] = tc_alg_fs

            t0 = time.time()
            obj_iter_ds, snr_iter_ds, rate_iter_ds, _ = Alg_PGD(Hii.numpy(), Gii.numpy(), F_ii.numpy(), W_ii.numpy(), delta, Pt.numpy(), smoothing_factor.detach().numpy(), NoisePower, Iter=200,
                                                                tolerance=1e-4, pgd_lr=step_size_fixed, BTS=True)
            t1 = time.time()
            tc_alg_ds = t1 - t0
            TC_algo_ds_all[ii, k_id] = tc_alg_ds
            # obj_spawc_all.append(obj_iter)
            obj_spawc_all_fs[ii, k_id] = obj_iter_fs
            SNR_spawc_all_fs[ii, k_id, :] = snr_iter_fs
            obj_spawc_all_ds[ii, k_id] = obj_iter_ds
            SNR_spawc_all_ds[ii, k_id, :] = snr_iter_ds

        t0 = time.time()
        loss_list, Obj_cache, Rate_cache, SNR_cache = myModel(H_batch[:, :, :K], G_batch, Gs_batch, F_batch, W_batch, Pt, NoisePower, delta, n_iter_W,n_iter_outer)
        t1 = time.time()
        tc_unfolding = t1 - t0
        TC_unfolding_all[k_id] = tc_unfolding

        obj_unfolding_all[:, k_id, :] = Obj_cache
        SNR_out = pow2db(np.mean(SNR_cache[:, -1, :].squeeze(), axis=0))
        SNR_avr[k_id, :] = SNR_out

        pass

    break
K_id = 1  # UE (K_id+1)*2
# channel_id = 1
obj_unfolding_smp = obj_unfolding_all[:, K_id, :]
obj_unfolding_avr = np.mean(obj_unfolding_smp, axis=0)
obj_alg_fs_all = obj_spawc_all_fs[:, K_id]
obj_alg_ds_all = obj_spawc_all_ds[:, K_id]

ccc = 1
obj_alg_fs_avr = average_padded_lists_1d(obj_alg_fs_all)
obj_alg_ds_avr = average_padded_lists_1d(obj_alg_ds_all)

plt.figure()
xx = list(range(len(obj_unfolding_avr)))
plt.plot(xx, obj_unfolding_avr, 'r-*', label='Unfolding')

xx = list(range(len(obj_alg_fs_avr)))
plt.plot(xx, obj_alg_fs_avr, 'b-s', label='Alg. 1 (fixed step size)')

xx = list(range(len(obj_alg_ds_avr)))
plt.plot(xx, obj_alg_ds_avr, 'k-s', label='Alg. 1 (dynamic step size)')

plt.xlabel('Number of iterations')
plt.ylabel('Objective value')
plt.legend(prop={'size': 10, 'weight': 'bold'})
plt.grid(True)
plt.box(True)
fig_name = 'Convergence_IterW'+str(n_iter_W)+'_IterOuter'+str(n_iter_outer)+'.png'
fig_path = directory_model + fig_name
plt.savefig(fig_path)  # save figure
plt.show()

TC_algo_fs_avr = np.mean(TC_algo_fs_all, axis=0)
TC_algo_ds_avr = np.mean(TC_algo_ds_all, axis=0)
TC_unfolding_avr = TC_unfolding_all / test_size

xx = np.array(list(range(2, Kmax + 1, 2)))
width = 0.5  # width of the bars
plt.figure()
plt.bar(xx, TC_algo_fs_avr, width=width, color='b', label='Alg.1 (fixed step size)')
plt.bar(xx + width, TC_algo_ds_avr, width=width, color='g', label='Alg.1 (dynamic step size)')
plt.bar(xx + 2 * width, TC_unfolding_avr, width=width, color='r', label='Unfolding')
plt.xlabel('Number of communication users')
plt.ylabel('Average time cost [s]')
plt.xticks(xx + width, xx)  # Set the x-ticks to be centered
plt.legend()
# plt.grid(True)
plt.box(True)
fig_name = 'TCvsUE'+'_IterW'+str(n_iter_W)+'_IterOuter'+str(n_iter_outer)+'.png'
fig_path = directory_model + fig_name
plt.savefig(fig_path)  # save figure
plt.show()

SNR_alg_fs = np.mean(SNR_spawc_all_fs, axis=0)
SNR_alg_ds = np.mean(SNR_spawc_all_ds, axis=0)
xx = list(range(2, Kmax + 1,2))
plt.figure()
plt.plot(xx, SNR_avr[:, 0], 'b-s', label='SINR (unfolding)', markersize=5)
plt.plot(xx, SNR_avr[:, 1], 'k-o', label='SCNR (unfolding)', markersize=5)
plt.plot(xx, SNR_alg_fs[:, 0], 'b--x', label='SINR (Alg. 1 with fixed step size)', markersize=5)
plt.plot(xx, SNR_alg_fs[:, 1], 'k--d', label='SCNR (Alg. 1 with fixed step size)', markersize=5)
plt.plot(xx, SNR_alg_ds[:, 0], 'b:^', label='SINR (Alg. 1 with dynamic step size)', markersize=5)
plt.plot(xx, SNR_alg_ds[:, 1], 'k:*', label='SCNR (Alg. 1 with dynamic step size)', markersize=5)
plt.xlabel('Number of communication users')
plt.ylabel('min SINR or min SCNR [dB]')
plt.legend(prop={'size': 10, 'weight': 'bold'})
plt.grid(True)
plt.box(True)
fig_name = 'SNRvsUE'+'_IterW'+str(n_iter_W)+'_IterOuter'+str(n_iter_outer)+'.png'
fig_path = directory_model + fig_name
plt.savefig(fig_path)  # save figure
plt.show()

data = {
    'Converg_unfolding': obj_unfolding_avr,
    'Converg_Alg_fs': obj_alg_fs_avr,
    'Converg_Alg_ds': obj_alg_ds_avr,
    'TC_Alg_fs': TC_algo_fs_avr,
    'TC_Alg_ds': TC_algo_ds_avr,
    'TC_Unfolding': TC_unfolding_avr,
    'SNR_Alg_fs': SNR_alg_fs,
    'SNR_Alg_ds': SNR_alg_ds,
    'SNR_Unfolding': SNR_avr,
    'delta': delta,
    'n_iter_W': n_iter_W,
    'n_iter_outer': n_iter_outer
    }

# Save the data to a .mat file
file_name = 'data_delta'+str(delta)+'.mat'
scipy.io.savemat(directory_model+file_name, data)
pass
t_end = time.time()
time_cost = (t_end - t_start)/60
print(f'---------End testing------time cost: {time_cost:.4f}min',flush=True)
