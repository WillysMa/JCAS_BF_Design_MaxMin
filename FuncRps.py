# -* coding: utf-8 -*-
'''
@File；FuncRps.py
@Author: Mengyuan Ma
@Contact: mamengyuan410@gmail.com
@Time: 2022-08-15 15:45
'''

import numpy as np
import torch
import os
# import torch.linalg

# from SysParams import *
# import math
import torch.nn as nn
import scipy
from scipy.linalg import eig
# from scipy.io import savemat
# from scipy.sparse.linalg import eigsh


from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def db2pow(db):
    """ Convert dB to power """
    return 10 ** (db / 10)

def pow2db(x):
    return 10 * np.log10(x)

def square_abs(x):
    """Calculate the absolute square of x using PyTorch."""
    # Check for NaN or inf values
    if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
        raise ValueError("Input contains NaN or infinity values.")

    # Compute the squared magnitude
    result = x * torch.conj(x)

    # Handle possible invalid values in the result
    if torch.any(torch.isnan(result)) or torch.any(torch.isinf(result)):
        raise RuntimeWarning("Result contains NaN or infinity values.")

    return torch.real(result)


def square_fnorm(A):
    """Calculate the squared Frobenius norm of matrix A using PyTorch."""
    # Ensure A is a complex tensor if required
    if not torch.is_complex(A):
        A = A.to(torch.complex128)  # Convert to complex if not already

    # Calculate the Hermitian transpose of A
    A_conj_transpose = torch.conj(A.T)

    # Compute the product A * A'
    product = torch.mm(A, A_conj_transpose)

    # Calculate the trace of the product
    out = torch.trace(product)

    return out.real  # Return as a scalar value

def calculateT_s(W, G, F, noise_s):
    """
    Calculate the quantity based on the batched matrices W, G, F and noise_s.

    Parameters:
    W (torch.Tensor): Matrix W (complex tensor) of shape (batch_size, Nt, K)
    G (torch.Tensor): Matrix G (complex tensor) of shape (batch_size, (M+C)*Nr, Nt)
    F (torch.Tensor): Matrix F (complex tensor) of shape (batch_size, Nr, M)
    noise_s (torch.Tensor): Noise level (scalar or tensor of shape [batch_size])

    Returns:
    torch.Tensor: Result of the calculation, shape (batch_size, M)
    """
    batch_size, Nr, M = F.shape
    _, Nt, K = W.shape
    T = G.shape[1] // Nr
    out = torch.zeros(batch_size, M, dtype=torch.float64)  # Output tensor

    # Ensure matrices are in complex128
    W = W.to(torch.complex128)
    G = G.to(torch.complex128)
    F = F.to(torch.complex128)
    # noise_s = noise_s.to(torch.float64)  # Ensure noise_s is float64

    Wtemp = torch.matmul(W, torch.conj(W.transpose(1, 2)))  # Shape: [batch_size, Nt, Nt]

    for m in range(M):
        fm = F[:, :, m]  # Shape: [batch_size, Nr]
        fm = fm.unsqueeze(2)  # Shape: [batch_size, Nr, 1]

        sum_value = torch.zeros(batch_size, dtype=torch.float64)

        for j in range(T):
            G_sub = G[:, Nr * j:Nr * (j + 1), :]  # Shape: [batch_size, Nr, Nt]
            temp = torch.matmul(torch.conj(fm.transpose(1, 2)), torch.matmul(G_sub, torch.matmul(Wtemp, torch.matmul(
                torch.conj(G_sub.transpose(1, 2)), fm))))  # Shape: [batch_size, 1, 1]
            sum_value += torch.real(temp).squeeze() # Sum over the batch dimension


        tmp_term = Nr * torch.sum(fm * torch.conj(fm), dim=1) * noise_s
        out[:, m] = sum_value + tmp_term.real.squeeze()

    return out

def calculate_scnr(T_sm, G_s, W, F):
    """
    Calculate the Signal-to-Noise-plus-Interference Ratio (SCNR) using PyTorch for batch data.

    Parameters:
    T_sm (torch.Tensor): Tensor representing the total signal minus interference, shape (batch_size, M).
    G_s (torch.Tensor): 4D tensor of channel matrices for each filter, shape (batch_size, Nr, Nt, M).
    W (torch.Tensor): 3D tensor for beamforming, shape (batch_size, Nt, K).
    F (torch.Tensor): 2D tensor of filters, shape (batch_size, Nr, M).

    Returns:
    torch.Tensor: SCNR for each filter, shape (batch_size, C).
    """
    batch_size, Nr, M = F.shape
    SCNR = torch.zeros(batch_size, M, dtype=torch.float64)

    # Compute W * W^H for each batch
    Wtemp = torch.matmul(W, W.conj().transpose(1, 2))  # Shape: (batch_size, Nt, Nt)

    # Compute SCNR for each batch and filter
    for m in range(M):
        fm = F[:, :, m].unsqueeze(2)  # Shape: (batch_size, Nr, 1)
        G_s_m = G_s[:, :, :, m]  # Shape: (batch_size, Nr, Nt)

        # Compute term = (fm^H * G_s_m * Wtemp * G_s_m^H * fm)
        term = torch.matmul(torch.conj(fm).transpose(1, 2), G_s_m)
        term = torch.matmul(term, Wtemp)
        term = torch.matmul(term, torch.conj(G_s_m).transpose(1, 2))
        term = torch.matmul(term, fm)

        # Compute SCNR
        SCNR[:, m] = torch.real(term).squeeze() ## trace operation are needed

    # Compute SCNR = SCNR / (T_sm - SCNR)
    SCNR = SCNR / (T_sm - SCNR)
    return SCNR


def logsumexp(mu, x):
    """
    Compute the logarithm of the sum of exponentials of input elements.

    Parameters:
    mu (float): A scaling factor for the exponentials.
    x (torch.Tensor): Input tensor of values.
    compute_softmax (bool): If True, also compute and return the softmax of the input values.

    Returns:
    tuple: A tuple with the following elements:
        - lse (float): The logarithm of the sum of exponentials.
        - sm (torch.Tensor, optional): Softmax of the input values, if requested.
    """
    if x.dim() != 1:
        raise ValueError("Input x must be a one-dimension array.")

    # Ensure x is a float tensor
    x = x.to(dtype=torch.float64)

    # Calculate the maximum value for numerical stability
    xmax = torch.max(x)

    # Compute the exponentials with the scaling factor
    exp_x = torch.exp(-mu * (x - xmax))

    # Sum of exponentials
    sum_exp = torch.sum(exp_x)

    # Log sum of exponentials
    lse = xmax + torch.log1p(sum_exp - 1)


    softmax = exp_x / (1 + sum_exp)
    return softmax


def calculate_beta(T_sm, G_s, W, alpha_sm, F):
    """
    Calculate the matrix output based on the provided inputs for batch data.

    Parameters:
    T_sm (torch.Tensor): Tensor of shape (batch_size, M) representing the total signal minus interference.
    G_s (torch.Tensor): 4D tensor of shape (batch_size, Nr, Nt, M) for channel matrices.
    W (torch.Tensor): 3D tensor of shape (batch_size, Nt, K) for beamforming matrices.
    alpha_sm (torch.Tensor): Tensor of shape (batch_size, M) for the parameters.
    F (torch.Tensor): 3D tensor of shape (batch_size, Nr, M) for filters.

    Returns:
    torch.Tensor: 3D tensor of shape (batch_size, M, K) representing the computed output.
    """
    batch_size, Nr, M = F.shape
    _, _, K = W.shape
    # Initialize the output tensor
    out = torch.zeros((batch_size, M, K), dtype=torch.complex128)

    # Ensure all tensors are complex
    W = W.to(torch.complex128)
    G_s = G_s.to(torch.complex128)
    F = F.to(torch.complex128)

    # Reshape tensors for broadcasting
    F_unsqueezed = F.unsqueeze(3)  # Shape: (batch_size, Nr, M, 1)
    sqrt_alpha = torch.sqrt(1 + alpha_sm).to(dtype=torch.complex128)  # Shape: (batch_size, M)
    # sqrt_alpha = sqrt_alpha.unsqueeze(2)  # Shape: (batch_size, M, 1)

    # Calculate the output for each batch
    for m in range(M):
        sqrt_term = sqrt_alpha[:, m]  # Shape: (batch_size, )
        fm = F_unsqueezed[:, :, m, :]  # Shape: (batch_size, Nr, 1)

        # Compute each row of `out` using batch matrix operations
        term = torch.conj(fm).transpose(1, 2) @ G_s[:, :, :, m] @ W  # Shape: (batch_size, 1, K)
        term = torch.diag(sqrt_term) @ term.squeeze()  # Shape: (batch_size, K)
        out[:, m, :] = (term / T_sm[:, m].unsqueeze(1)).squeeze(1)  # Shape: (batch_size, K)

    return out

def calculate_F(F, theta_sm, xi_sm, G_s, G, W, NoisePower):
    """
    Compute F_Last and F based on batch data.

    Parameters:
    F (torch.Tensor): Tensor of filters (batch_size, Nr, M).
    theta_sm (torch.Tensor): Tensor of size (batch_size, M, K) representing theta_sm.
    xi_sm (torch.Tensor): Tensor of size (batch_size, M) representing xi_sm.
    G_s (torch.Tensor): Tensor of size (batch_size, Nr, Nt, M) representing channel matrices.
    G (torch.Tensor): Tensor of size (batch_size, (M+C)*Nr, Nt) representing channel matrices.
    W (torch.Tensor): Tensor of size (batch_size, Nt, K) representing W.
    NoisePower (float): Noise level.
    C (int): Number of clutters.

    Returns:
    torch.Tensor: Updated F tensor.
    """
    batch_size, Nr, M = F.shape
    MplusC = G.shape[1] // Nr
    # Initialize F_Last
    F_Last = torch.zeros_like(F, dtype=torch.complex128, device=F.device)

    # Compute F_Last
    for m in range(M):
        thetam = theta_sm[:, m, :].unsqueeze(2)  # Shape (batch_size, K, 1)
        term = torch.sqrt(1 + xi_sm[:, m]) / (thetam.transpose(1, 2) @ thetam.conj()).squeeze()
        middle = G_s[:, :, :, m] @ W @ thetam.conj()
        F_Last[:, :, m] = torch.diag(term) @ middle.squeeze()

    # Compute temp
    temp = 0
    WWtmp = W @ W.conj().transpose(1, 2)
    for j in range(MplusC):

        G_block = G[:, j * Nr:(j + 1) * Nr, :]  # Extract batch-wise block
        temp += G_block @ WWtmp @ G_block.conj().transpose(1, 2)

    # Compute F
    noise_part = torch.stack([torch.eye(Nr, dtype=torch.complex128, device=F.device) * NoisePower*Nr for i in range(batch_size)])

    # intrfn = torch.linalg.inv(temp + noise_part)
    F = torch.linalg.inv(temp + noise_part) @ F_Last

    ccc = 1
    return F

def calculateA_s(z_sm, xi_sm, theta_sm, G_s, F):
    """
    Calculate the matrix A_s based on the given inputs, supporting batch processing.

    Parameters:
    - z_sm (torch.Tensor): 2D tensor of shape (batch_size, M) where M is the number of rows in beta_sm.
    - xi_sm (torch.Tensor): 2D tensor of shape (batch_size, M) where M is the number of rows in beta_sm.
    - theta_sm (torch.Tensor): 3D tensor of shape (batch_size, M, K).
    - G_s (torch.Tensor): 4D tensor of shape (batch_size, Nr, Nt, M).
    - F (torch.Tensor): 3D tensor of shape (batch_size, Nr, M).

    Returns:
    - A_s (torch.Tensor): 3D tensor of shape (batch_size, K, Nt) representing the computed output.
    """
    batch_size, M, K = theta_sm.shape
    _, _, Nt, _ = G_s.shape

    # A_s = torch.zeros((batch_size, K, Nt), dtype=torch.complex128)
    A_s = 0

    for m in range(M):
        temp = z_sm[:, m] * torch.sqrt(1 + xi_sm[:, m])
        # Perform matrix operations
        theta_sm_m = torch.conj(theta_sm[:, m, :]) # Shape (batch_size, K)
        mid_term = theta_sm_m.unsqueeze(2) @ torch.conj(F[:, :, m]).unsqueeze(1) @ G_s[:, :, :, m] # Shape (batch_size, K, Nt)

        A_s += temp.unsqueeze(1).unsqueeze(2) * mid_term  # broadcasting along the first dimension, i.e. the batch_size

    return A_s


def calculateB_S(z_sm, theta_sm, G, F):
    """
    Calculate the matrix B_s based on the given inputs with batch support.

    Parameters:
    - z_sm (torch.Tensor): 2D tensor of shape (batch_size, M) where M is the number of rows in theta_sm.
    - theta_sm (torch.Tensor): 3D tensor of shape (batch_size, M, K).
    - G (torch.Tensor): 3D tensor of shape (batch_size, (N+C)*Nr, Nt).
    - F (torch.Tensor): 3D tensor of shape (batch_size, Nr, M).

    Returns:
    - B_s (torch.Tensor): 3D tensor of shape (batch_size, Nt, Nt).
    """
    batch_size, Nr, M = F.shape
    # Nt = G.shape[2]
    # B_s = torch.zeros((batch_size, Nt, Nt), dtype=torch.complex128, device=F.device)
    B_s = 0
    # Compute the sums vector for each batch
    sums = torch.zeros((batch_size, M), dtype=torch.complex128, device=F.device)
    for m in range(M):
        # Compute the sums vector for each batch
        theta_sm_m = theta_sm[:, m, :]  # shape (batch_size, K)
        temp = theta_sm_m.unsqueeze(1).conj() @ theta_sm_m.unsqueeze(2)
        sums[:, m] = z_sm[:, m]*temp.squeeze()
    # diag_sum = torch.diag_embed(sums)
    mid_term = F @ torch.diag_embed(sums) @ F.conj().transpose(1, 2)
    # Compute B_s for each batch
    num_blocks = G.shape[1] // Nr
    for j in range(num_blocks):
        G_block = G[:, Nr * j:Nr * (j + 1), :]   # shape of (batch_size, Nr, Nt)
        term = G_block.conj().transpose(1, 2) @ mid_term @ G_block
        B_s += term

    pass
    return B_s


def normalize_grad(grad):
    """
    Normalize the gradient tensor by its Frobenius norm for each matrix in the batch.

    Parameters:
    - grad (torch.Tensor): A 3D tensor of shape (batch_size, m, n) representing the gradients for each matrix in the batch.

    Returns:
    - torch.Tensor: A 3D tensor of the same shape as `grad` with each matrix normalized by its Frobenius norm.
    """
    # Compute Frobenius norm for each matrix in the batch
    norms = torch.norm(grad, p='fro', dim=(1, 2), keepdim=True)

    # Avoid division by zero by adding a small epsilon where norms are zero
    epsilon = 1e-8
    norms = torch.max(norms, torch.tensor(epsilon, dtype=norms.dtype, device=norms.device))

    # Normalize the gradients by their Frobenius norm
    normalized_grad = grad / norms

    return normalized_grad



def obj_func(W, B_t, A_t):
    """
    Calculate the objective function value.

    Parameters:
    - W (torch.Tensor): The matrix W.
    - B_t (torch.Tensor): The matrix B_t.
    - A_t (torch.Tensor): The matrix A_t.

    Returns:
    - float: The value of the objective function.
    """
    # Ensure all tensors are complex
    W = W.to(dtype=torch.complex128)
    B_t = B_t.to(dtype=torch.complex128)
    A_t = A_t.to(dtype=torch.complex128)

    # Compute the objective function
    W_conj_T = torch.conj(W.T)  # Conjugate transpose of W
    term1 = torch.trace(W @ W_conj_T @ B_t).real  # np.real is not needed in PyTorch
    term2 = 2 * torch.trace(W @ A_t).real  # np.real is not needed in PyTorch

    return term1 - term2


def optimize(W, B_t, A_t, Pt, Nt, max_iter=100, tol=1e-4):
    """
    Optimize the matrix W using the iterative algorithm.

    Parameters:
    - W (torch.Tensor): Initial matrix W.(Nt,K)
    - B_t (torch.Tensor): Matrix B_t. (Nt,Nt)
    - A_t (torch.Tensor): Matrix A_t. (K,Nt)
    - Pt (float): Scalar Pt value.
    - Nt (int): Scalar Nt value.
    - max_iter (int): Maximum number of iterations.
    - tol (float): Convergence tolerance.

    Returns:
    - torch.Tensor: Optimized matrix W.
    - list: List of objective function values over iterations.
    """
    obj = obj_func(W, B_t, A_t)
    objsub = []

    # Compute eigenvalues of B_t
    eigenvalues, _ = torch.linalg.eig(B_t)
    mu = abs(eigenvalues[0].item())  # Taking the magnitude of the largest eigenvalue

    for _ in range(max_iter):
        obj_old = obj

        # Update W
        W = torch.conj(A_t.T) + (mu * torch.eye(Nt, dtype=torch.complex128) - B_t) @ W
        diag_term = torch.sqrt(Pt / Nt / torch.diag(W @ torch.conj(W.T)))
        W = torch.diag(diag_term) @ W

        obj = obj_func(W, B_t, A_t)
        objsub.append(obj.item())

        accW = abs(obj - obj_old) / abs(obj_old)
        if accW <= tol:
            break

    return W, objsub


import torch


def proj_W(W, Pt):
    """
    Project each matrix in a batch of matrices W onto the feasible point by applying a diagonal scaling.

    Parameters:
    - W (torch.Tensor): A 3D tensor of shape (batch_size, Nt, K) representing a batch of matrices.
    - Pt (float): Scalar Pt value.

    Returns:
    - torch.Tensor: The projected batch of matrices with the same shape as W.
    """
    batch_size, Nt, K = W.shape

    # Compute W * W^H (Hermitian transpose) for each matrix in the batch
    W_W_H = W @ W.conj().transpose(1, 2)

    # Compute diagonal scaling matrix for each matrix in the batch
    diag_elements = Pt / Nt / torch.diagonal(W_W_H, 0, 1, 2)

    # Compute the scaling matrix for each matrix in the batch
    scaling_matrices = torch.diag_embed(torch.sqrt(diag_elements))

    # Apply the scaling matrix to each matrix in the batch
    W_projected = scaling_matrices @ W

    return W_projected





def PGD_method(X, B_t, A_t, Pt, tol=1e-4, iter_max=200, alpha=0.1, beta=0.3):
    """
    Perform Projected Gradient Descent (PGD) on a matrix F.

    Parameters:
    - F (torch.Tensor): Initial matrix.
    - tol (float): Tolerance for convergence.
    - iter_max (int): Maximum number of iterations.
    - alpha (float): Step size parameter.
    - beta (float): Backtracking line search parameter.

    Returns:
    - Obj_Opt (float): Optimal objective value.
    - Fopt (torch.Tensor): Optimal matrix F.
    - iter_data (list): List of objective values over iterations.
    """
    fw = obj_func(X, B_t, A_t)
    fw_all = []

    for _ in range(iter_max):
        X_old = X.clone()
        fw_old = fw

        grad = B_t @ X - A_t.conj().T
        grad_old = grad

        grad_unit = grad / torch.norm(grad, p='fro')

        step_size = 1
        counter = 0
        while counter <= 30:
            counter += 1
            NablaX = grad_unit
            x_new = X_old - step_size * NablaX
            x_proj = proj_W(x_new, Pt)
            obj_new = obj_func(x_proj, B_t, A_t)
            obj_benchmark = fw_old - alpha * step_size * torch.norm(grad_unit, p='fro') ** 2

            if obj_new < obj_benchmark:
                break
            else:
                step_size = beta * step_size

        X = X- step_size * grad_unit
        X = proj_W(X, Pt)
        fw = obj_func(X, B_t, A_t)

        grad_new = B_t @ X - A_t.conj().T
        fw_all.append(fw.item())
        err = torch.norm(grad_new - grad_old, p='fro') ** 2 / torch.norm(grad_old, p='fro') ** 2
        if err <= tol:
            break

    Xopt = X
    Obj_Opt = fw
    iter_data = fw_all

    # Plotting
    # plt.figure()
    # plt.plot(iter_data)
    # plt.ylabel('Objective')
    # plt.xlabel('Iteration')
    # plt.grid(True)
    # plt.box(True)
    # plt.title('PGD Algorithm')
    # plt.show()

    return Xopt



def channel(Nt, Nr, K, M, C, seed):
    # Constants
    noise_level = -80
    noise_k = noise_level  # dBm
    noise_k_norm = np.sqrt(db2pow(noise_k ))

    noise_s = noise_level  # dBm
    noise_s_norm = np.sqrt(db2pow(noise_s ))

    alpha_c = 3
    alpha_s = 2
    T_0 = -10

    np.random.seed(seed)

    # Distance calculations
    distance_c = 100 + 20 * np.random.randn(K)
    distance_s = 10 + 2 * np.random.randn(M + C)

    distance_c = np.sqrt(distance_c ** 2 + 10 ** 2)  # Pythagoras' theorem
    distance_s = np.sqrt(distance_s ** 2 + 10 ** 2)

    beta_c = np.sqrt(db2pow(T_0) * (distance_c ** (-alpha_c)))
    beta_s = np.sqrt(db2pow(T_0) * (distance_s ** (-alpha_s)))

    angle_all = -np.pi / 3 + (4 * np.pi / 3) * np.random.rand(K)

    K_a = 5
    Hall = np.zeros((Nt, K), dtype=np.complex64)
    for iter in range(K):
        at = (1 / np.sqrt(Nt)) * np.exp(1j * np.pi * np.sin(angle_all[iter]) * np.arange(Nt))
        NLos = (1 / np.sqrt(2)) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
        NLos /= np.linalg.norm(NLos)
        Hall[:, iter] = np.sqrt(K_a / (K_a + 1)) * at + np.sqrt(1 / (K_a + 1)) * NLos
    Hall = Hall @ np.diag(beta_c) / noise_k_norm

    angle_all = -np.pi / 3 + (4 * np.pi / 3) * np.random.rand(M + C)

    Gall = []

    for iter in range(M + C):
        ar = (1 / np.sqrt(Nr)) * np.exp(1j * np.pi * np.sin(angle_all[iter]) * np.arange(Nr))
        at = (1 / np.sqrt(Nt)) * np.exp(1j * np.pi * np.sin(angle_all[iter]) * np.arange(Nt))
        if iter < M:
            G_s = beta_s[iter] ** 2 * np.outer(ar, at.conj()) / noise_s_norm
            Gall.append(G_s)
        else:
            G_c = beta_s[iter - M] ** 2 * np.outer(ar, at.conj()) / noise_s_norm
            Gall.append(G_c)

    Gall = np.concatenate(Gall, axis=0)

    ccc=1
    return Hall, Gall


def Initialize_WnF(H, G, Pt, Nr, Nt, M, C, noise_s):
    """
    Initialize W and F.

    Parameters:
    - H (torch.Tensor): channel matrix of  shape ( Nt, K).
    - G (torch.Tensor): sensing and clutter channel matrix of  shape ( (M+C)*Nr, Nt).
    - Pt (float): Scalar Pt value.
    - Nr (float): Scalar Nr value.
    - Nt (float): Scalar Nt value.
    - M (float): Scalar M value.
    - C (float): Scalar C value.
    - noise_s (float): Scalar noise_s value.

    Returns:
    - torch.Tensor: W of shape (Nt,K) and F of shape (Nr,M).
    """

    M_plus_C = M + C

    # Initialize tensors
    G_s = torch.zeros((Nr, Nt, M), dtype=torch.complex128)
    # G_c = torch.zeros((Nr, Nt, C), dtype=torch.complex128)

    for iter in range(M):
        G_s[:, :, iter] = G[(iter * Nr):(iter + 1) * Nr, :]


    # Initialize W
    # P_k = Pt / M

    # W = H @ torch.linalg.inv(H.conj().T @ H)

    W = H  # MRT precoder


    Wtemp = W @ W.conj().T
    W = torch.diag(torch.sqrt(Pt / Nt / torch.diag(Wtemp))) @ W  # Project onto feasible point

    # Initialize F
    F = torch.eye(Nr, M, dtype=torch.complex128)
    Wtemp = W @ W.conj().T
    for m in range(M):
        temp = torch.eye(Nr, dtype=torch.complex128) * noise_s
        for j in range(M_plus_C):
            if j != m:
                G_block = G[(j * Nr):(j + 1) * Nr, :]
                temp += G_block @ Wtemp @ G_block.conj().T

        A = G[(m * Nr):(m + 1) * Nr, :] @ Wtemp @ G[(m * Nr):(m + 1) * Nr, :].conj().T

        # Solve generalized eigenvalue problem
        A_inv = torch.linalg.inv(A)
        Atmp = A_inv @ temp

        eigvals, eigvecs = torch.linalg.eig(Atmp)
        # Find the index of the maximum eigenvalue
        index = torch.argmax(square_abs(eigvals))
        F[:, m] = torch.real(eigvecs[:, index])
    pass
    return G_s, W, F

def GenChannel(Num, Nr, Nt, K, M, C, data_path='./'):

    Hall = np.zeros((Num, Nt, K), dtype=np.complex128)
    Gall = np.zeros((Num, (M+C)*Nr, Nt), dtype=np.complex128)

    for i in range(Num):
        Hi, Gi = channel(Nt, Nr, K, M, C, seed=i)

        Hall[i, :, :] = Hi
        Gall[i, :, :] = Gi

    np.savez(data_path, Hall=Hall, Gall=Gall)

    # if test:
    #     data = {
    #         'Hall': Hall,
    #         'Gall': Gall
    #     }
    #
    #     # Save the data to a .mat file
    #     scipy.io.savemat('test_data.mat', data)



class CustomDataset(Dataset):
    def __init__(self, file_path):
        data = np.load(file_path)
        self.Hall = torch.tensor(data['Hall'])
        self.Gall = torch.tensor(data['Gall'])

    def __len__(self):
        return len(self.Hall[:, 0, 0])

    def __getitem__(self, idx):
        data = {'Hall':self.Hall[idx, :, :], 'Gall': self.Gall[idx,:,:]}
        return data


class LogInfos():
    def __init__(self, file_directory='./', para_file_name='Logs_Info.txt'):
        self.dir_para_file = os.path.join(file_directory, para_file_name)
        file_para = open(self.dir_para_file, 'w')
        file_para.write('System parameters:\n')
        file_para.close()

        pass

    def add_logs(self, infos):
        if isinstance(infos, str):
            file_para = open(self.dir_para_file, 'a')
            file_para.write('\n')
            file_para.write(infos + '\n')
            file_para.close()
        else:
            raise TypeError(f"Invalid input type: {type(infos).__name__}. Expected a string.")
        pass


class UnfoldNet(torch.nn.Module):
    def __init__(self, smoothing_factor, step_size, Nr, Nt, K, M, C,NoisePower, delta):
        super().__init__()
        self.NumIter_outer, self.NumIter_inter = step_size.shape
        self.step_size = nn.Parameter(step_size)
        self.smoothing_factor = nn.Parameter(smoothing_factor)
        # self.smoothing_factor = smoothing_factor
        self.Nr, self.Nt, self.K, self.M, self.C, self.NoisePower, self.delta = Nr, Nt, K, M, C, NoisePower, delta

        # self.decay_rate = nn.Parameter(torch.ones(self.NumIter_outer)*0.6)
        # self.layer1 = nn.Sequential(
        #     nn.Linear(self.K, self.K, dtype=torch.float64),
        #     nn.BatchNorm1d(self.K, momentum=0.2, dtype=torch.float64),
        #     nn.PReLU(dtype=torch.float64),
        #     nn.Linear(self.K, self.K, dtype=torch.float64),
        #     nn.BatchNorm1d(self.K, momentum=0.2, dtype=torch.float64),
        #     nn.Sigmoid()
        # )
        #
        # self.layer2 = nn.Sequential(
        #     nn.Linear(self.M, self.M, dtype=torch.float64),
        #     nn.BatchNorm1d(self.M, momentum=0.2, dtype=torch.float64),
        #     nn.PReLU(dtype=torch.float64),
        #     nn.Linear(self.M, self.M, dtype=torch.float64),
        #     nn.BatchNorm1d(self.M, momentum=0.2, dtype=torch.float64),
        #     nn.Sigmoid()
        # )

    def forward(self, H, G, G_s, F, W, Pt, NoisePower, delta, n_iter_W, NumIter_outer):
        batch_size, _ , _ = H.shape
        Obj_cache = np.zeros([batch_size, NumIter_outer])
        SNR_cache = np.zeros([batch_size, NumIter_outer, 2])
        Rate_cache = np.zeros([batch_size, NumIter_outer, 2])

        Loss = []
        for iter_out in range(NumIter_outer):
            HW_product = torch.matmul(H.conj().transpose(1, 2), W)
            T_ck = torch.sum(torch.abs(HW_product) ** 2, dim=2) + NoisePower
            T_s = calculateT_s(W, G, F, NoisePower)

            # HW_diag_elements = torch.stack( [torch.diag(HW_product[i,:,:]) for i in range(batch_size)])
            HW_diag_elements = torch.diagonal(HW_product, 0, 1, 2)
            xi_ck = T_ck / (T_ck - torch.abs(HW_diag_elements) ** 2) - 1
            xi_sm = calculate_scnr(T_s, G_s, W, F)

            # z_ck = self.layer1(xi_ck)
            # z_ck = z_ck / torch.sum(z_ck, dim=1, keepdim=True)
            # z_sm = self.layer2(xi_sm)
            # z_sm = z_sm / torch.sum(z_sm, dim=1, keepdim=True)

            z_ck = torch.stack([logsumexp(self.smoothing_factor[0], torch.log1p(xi_ck[i, :])) for i in range(batch_size)])
            z_sm = torch.stack([logsumexp(self.smoothing_factor[1], torch.log1p(xi_sm[i, :])) for i in range(batch_size)])

            theta_ck = torch.sqrt(1 + xi_ck) * HW_diag_elements / T_ck
            theta_sm = calculate_beta(T_s, G_s, W, xi_sm, F)

            F = calculate_F(F, theta_sm, xi_sm, G_s, G, W,  self.NoisePower)  # update F

            diag_matrix = torch.diag_embed(z_ck * torch.sqrt(1 + xi_ck) * theta_ck.conj())
            # diag_matrix = torch.stack( [ torch.diag(diag_elements[i, :]) for i in range(batch_size) ])

            A_c = diag_matrix @ torch.conj(H.transpose(1, 2))
            X = calculateA_s(z_sm, xi_sm, theta_sm, G_s, F)

            # middle_term_v1 = torch.stack( [ torch.diag(z_ck[i, :] * torch.abs(theta_ck[i, :]) ** 2) for i in range(batch_size) ])
            middle_term = torch.diag_embed(z_ck * torch.abs(theta_ck) ** 2)
            B_c = H @ middle_term.to(dtype=torch.complex128) @ torch.conj(H.transpose(1, 2))
            Y = calculateB_S(z_sm, theta_sm, G, F)

            A_t = A_c + delta * X
            B_t = B_c + delta * Y

            grad = B_t @ W - A_t.conj().transpose(1, 2)
            grad_unit = normalize_grad(grad)
            for update_w in range(n_iter_W):
                for iter_inter in range(self.NumIter_inter):
                    W_new = W - (self.step_size[iter_out, iter_inter])* grad_unit
                    # W_new = W - self.step_size[iter_inter] * grad_unit
                    W = proj_W(W_new, Pt)

            WWtemp = W @ W.conj().transpose(1, 2)
            trace_term1 = torch.stack([torch.trace(WWtemp[i, :, :] @ B_t[i, :, :]).real for i in range(batch_size)])
            trace_term2 = torch.stack([torch.trace(W[i, :, :] @ A_t[i, :, :]).real for i in range(batch_size)])
            obj_pgd = torch.mean(trace_term1 - 2 * trace_term2)


            rate_sm, _ = torch.min(torch.log1p(xi_sm), 1)
            rate_ck, _ = torch.min(torch.log1p(xi_ck), 1)
            obj_batch = delta * rate_sm + rate_ck
            Obj_cache[:, iter_out] = obj_batch.detach().numpy()

            obj_mean = torch.mean(obj_batch)   # Store objective value for each iteration

            # temp_stack = torch.stack((rate_ck, rate_sm),1)
            Rate_cache[:, iter_out, :] = torch.stack((rate_ck.detach(), rate_sm.detach()),1).numpy()

            snr_ck, _ = torch.min(xi_ck, 1)
            snr_sm, _ = torch.min(xi_sm, 1)
            temp_stack = torch.stack((snr_ck.detach(), snr_sm.detach()),1).numpy()
            SNR_cache[:, iter_out, :] = torch.stack((snr_ck.detach(), snr_sm.detach()),1).numpy()

            scalar = 10
            alpha = 1.5
            episilon = 0.1
            # Loss.append( -np.log(iter_out + 1) * obj_mean) ->not good as the below two
            Loss.append(-1/(iter_out+ 1) * obj_mean)
            # Loss.append(1/(iter_out + 1) * obj_pgd)
            # Loss.append(-obj_mean) # not good as the above two
            # Loss.append(obj_pgd)
        pass
        return Loss, Obj_cache, Rate_cache, SNR_cache



def pad_list(lst, target_length):
    """Pad a list with its last value to achieve the target length."""
    if len(lst) < target_length:
        last_value = lst[-1]
        padded_list = lst + [last_value] * (target_length - len(lst))
    else:
        padded_list = lst
    return padded_list


def average_padded_lists_1d(lists_1d):
    """Compute the average of padded lists across a 1D array of lists."""
    # Flatten the 1D array to a list of lists
    flattened_lists = lists_1d

    # Find the maximum length among all lists
    max_length = max(len(lst) for lst in flattened_lists)

    # Pad all lists to the maximum length
    padded_lists = [pad_list(lst, max_length) for lst in flattened_lists]

    # Convert to numpy array for easier averaging
    padded_array = np.array(padded_lists)

    # Compute the average along each dimension
    average_array = np.mean(padded_array, axis=0)

    return average_array


if __name__ == '__main__':
    # SNR_dB_train = 20
    # Pt = 10 ** (SNR_dB_train / 10)
    # Nt = 16
    # Nr = 8
    # K = 4  # users
    # M = 2  # sensing targets
    # C = 2  # clutters
    # NoisePower = 1.
    # func_test(Pt, Nr, Nt, M, C, NoisePower)

    # Nt = 16
    # Nr = 16
    # K = 4  # users
    # M = 2  # sensing targets
    # C = 2  # clutters
    # GenChannel(train_size, Nr, Nt, K, M, C, data_path=data_path_train)

    ccc=1
