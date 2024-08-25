# -*- coding: utf-8 -*-
'''
# @File    : Alg_Numpy.py
# @Author  : Mengyuan Ma
# @Contact: mamengyuan410@gmail.com
# @Time    : 2024/7/31 20:26
'''

from SysParams import *
import numpy as np
import scipy
import matplotlib.pyplot as plt

def square_abs(x):
    """Calculate the absolute square of x."""
    # Check for NaN or inf values
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Input contains NaN or infinity values.")

    # Compute the squared magnitude
    result = x * np.conj(x)

    # Handle possible invalid values in the result
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        raise RuntimeWarning("Result contains NaN or infinity values.")

    return np.real(result)

def square_fnorm(A):
    """Calculate the squared Frobenius norm of matrix A."""
    # Calculate the Hermitian transpose of A

    # Compute the product A * A'
    product = np.dot(A, A.T.conj())

    # Calculate the trace of the product
    out = np.trace(product)

    return out.real

def db2pow(db):
    """ Convert dB to power """
    return 10 ** (db / 10)

def pow2db(x):
    return 10 * np.log10(x)

def calculateT_s(W, G, F, noise_s):
    """
    Calculate the quantity based on the matrices W, G, F and noise_s.

    Parameters:
    W (numpy.ndarray): Matrix W
    G (numpy.ndarray): Matrix G
    F (numpy.ndarray): Matrix F
    noise_s (float): Noise level

    Returns:
    numpy.ndarray: Result of the calculation
    """
    Nr, M = F.shape
    T = G.shape[0] // Nr
    out = np.zeros(M)

    Wtemp = W @ W.conj().T

    for m in range(M):
        sum_value = 0
        fm = F[:, m][:,np.newaxis]
        for j in range(T):
            G_sub = G[Nr * (j):Nr * (j + 1), :]
            temp = fm.conj().T @ G_sub @ Wtemp @ G_sub.conj().T @ fm
            sum_value += np.real(np.trace(temp))
        out[m] = sum_value + Nr * square_fnorm(fm) * noise_s

    return out


def calculate_scnr(T_sm, G_s, W, F):
    """
    Calculate the Signal-to-Noise-plus-Interference Ratio (SCNR).

    Parameters:
    T_sm (numpy.ndarray): Matrix representing the total signal minus interference.
    G_s (numpy.ndarray): 3D array of channel matrices for each filter.
    W (numpy.ndarray): Matrix for beamforming.
    F (numpy.ndarray): Matrix of filters.

    Returns:
    numpy.ndarray: SCNR for each filter.
    """
    _, M = F.shape
    SCNR = np.zeros(M)
    Wtemp = W @ W.conj().T
    for m in range(M):
        fm = F[:, m][:, np.newaxis]
        G_s_m = G_s[:, :, m]  # Extract the m-th matrix from G_s
        term = fm.conj().T @ G_s_m @ Wtemp @ G_s_m.conj().T @ fm
        SCNR[m] = np.real(np.trace(term))

    SCNR = SCNR / (T_sm - SCNR)
    return SCNR

def logsumexp(mu, x):
    """
    Compute the logarithm of the sum of exponentials of input elements.

    Parameters:
    mu (float): A scaling factor for the exponentials.
    x (numpy.ndarray): Input array of values.

    Returns:
    tuple: A tuple with the following elements:
        - lse (float): The logarithm of the sum of exponentials.
        - sm (numpy.ndarray): Softmax of the input values, if requested.
    """
    if not np.ndim(x) == 1:
        raise ValueError("Input x must be a vector.")

    n = len(x)
    s = 0
    e = np.zeros(n)

    xmax = np.min(x)
    a = xmax

    # Compute e and s
    for i in range(n):
        e[i] = np.exp(-mu * (x[i] - xmax))
        if i != np.argmin(x):
            s += e[i]

    lse = a + np.log1p(s)

    # If the softmax is required, compute it
    # if 'sm' in locals():
    sm = e / (1 + s)

    return sm


def calculate_beta(T_sm, G_s, W, alpha_sm, F):
    """
    Calculate the matrix out based on the provided inputs.

    Parameters:
    T_sm (numpy.ndarray): 1D array of size M.
    G_s (numpy.ndarray): 3D array of shape (Nr, Nt, M).
    W (numpy.ndarray): 2D array of shape (Nt, K).
    alpha_sm (numpy.ndarray): 1D array of size M.
    F (numpy.ndarray): 2D array of shape (Nr, M).

    Returns:
    numpy.ndarray: 2D array of shape (M, K) representing the computed output.
    """
    Nr, Nt, M = G_s.shape
    _, K = W.shape
    out = np.zeros((M, K), dtype=np.complex64)  # Ensure output has appropriate type

    for m in range(M):
        # Compute each row of `out` using the provided formula
        sqrt_term = np.sqrt(1 + alpha_sm[m])
        out[m, :] = (sqrt_term * np.conj(F[:, m]).T @ G_s[:, :, m] @ W) / T_sm[m]

    return out



def calculateA_s(y_sm, alpha_sm, beta_sm, G_s, F):
    """
    Calculate the matrix A_s based on the given inputs.

    Parameters:
    - y_sm (numpy.ndarray): 1D array of shape (M,) where M is the number of rows in beta_sm.
    - alpha_sm (numpy.ndarray): 1D array of shape (M,) where M is the number of rows in beta_sm.
    - beta_sm (numpy.ndarray): 2D array of shape (M, K).
    - G_s (numpy.ndarray): 3D array of shape (N, Nt, M).
    - F (numpy.ndarray): 2D array of shape (N, M).

    Returns:
    - A_s (numpy.ndarray): 2D array of shape (K, Nt).
    """
    _, Nt, _ = G_s.shape
    M, K = beta_sm.shape

    A_s = np.zeros((K, Nt), dtype=np.complex64)

    for m in range(M):

        beta_sm_m = beta_sm[m, :].conj().T

        # Perform matrix operations
        term = y_sm[m] * np.sqrt(1 + alpha_sm[m]) * np.outer(beta_sm_m, F[:, m].T.conj()) @ G_s[:, :, m]  # debug
        A_s += term

    return A_s



def calculateB_S(y_sm, beta_sm, G, F):
    """
    Calculate the matrix B_s based on the given inputs.

    Parameters:
    - y_sm (numpy.ndarray): 1D array of shape (M,) where M is the number of rows in beta_sm.
    - beta_sm (numpy.ndarray): 2D array of shape (M, K).
    - G (numpy.ndarray): 2D array of shape (N, Nt).
    - F (numpy.ndarray): 2D array of shape (Nr, M).

    Returns:
    - B_s (numpy.ndarray): 2D array of shape (Nt, Nt).
    """
    Nr, M = F.shape
    sums = np.zeros(M, dtype=complex)
    Nt = G.shape[1]
    B_s = np.zeros((Nt, Nt), dtype=complex)

    # Compute the sums vector
    for m in range(M):
        sums[m] = y_sm[m] * np.trace(np.outer(beta_sm[m, :], beta_sm[m, :].conj()))

    # Compute B_s
    num_blocks = G.shape[0] // Nr
    for j in range(num_blocks):
        G_block = G[Nr * j:Nr * (j + 1), :]
        term = G_block.T.conj() @ F @ np.diag(sums) @ F.T.conj()  @ G_block
        B_s += term

    return B_s


def obj_func(W, B_t, A_t):
    """
    Calculate the objective function value.

    Parameters:
    - W (numpy.ndarray): The matrix W of shape (Nt, K).
    - B_t (numpy.ndarray): The matrix B_t of shape (Nt, Nt).
    - A_t (numpy.ndarray): The matrix A_t of shape (K, Nt).
    Returns:
    - float: The value of the objective function.
    """
    term1 = np.real(np.trace(W @ W.T.conj() @ B_t))
    term2 = 2 * np.real(np.trace(W @ A_t))
    return term1 - term2



def optimize(W, B_t, A_t, Pt, Nt, max_iter=100, tol=1e-4):
    """
    Optimize the matrix W using the iterative algorithm.

    Parameters:
    - W (numpy.ndarray): Initial matrix W.
    - B_t (numpy.ndarray): Matrix B_t.
    - A_t (numpy.ndarray): Matrix A_t.
    - Pt (float): Scalar Pt value.
    - mu (float): Scalar mu value.
    - Nt (int): Scalar Nt value.
    - max_iter (int): Maximum number of iterations.
    - tol (float): Convergence tolerance.

    Returns:
    - numpy.ndarray: Optimized matrix W.
    - list: List of objective function values over iterations.
    """
    obj = obj_func(W, B_t, A_t)
    objsub = []

    eigenvalues, _ = scipy.sparse.linalg.eigsh(B_t, k=1, which='LM')
    mu = abs(eigenvalues[0])
    for _ in range(max_iter):
        obj_old = obj
        # Update W
        W = A_t.T.conj() + (mu * np.eye(Nt) - B_t) @ W
        W = np.diag(  np.sqrt( Pt / Nt / np.diag(W @ W.T.conj()) )  ) @ W

        obj = obj_func(W, B_t, A_t)
        objsub.append(obj)

        accW = np.abs(obj - obj_old) / np.abs(obj_old)
        if accW <= tol:
            break
    # Plot the objective function values
    # plt.plot(obj_values)
    # plt.xlabel('Iteration')
    # plt.ylabel('Objective Function Value')
    # plt.title('Convergence of the Objective Function')
    # plt.grid(True)
    # plt.show()
    return W, objsub


def proj(W, Pt):
    """
    Project matrix W onto the feasible point by applying a diagonal scaling.

    Parameters:
    - W (torch.Tensor): The matrix to be projected.
    - Pt (float): Scalar Pt value.
    - Nt (int): Scalar Nt value.

    Returns:
    - torch.Tensor: The projected matrix.
    """
    # Compute W * W^H (Hermitian transpose)
    W_W_H = W @ W.conj().T

    Nt, _ = W.shape
    # Compute diagonal scaling matrix
    diag_elements = np.sqrt(Pt / Nt / np.diag(W_W_H))
    scaling_matrix = np.diag(diag_elements)

    # Apply the scaling matrix to W
    return scaling_matrix @ W

def PGD_method(X, B_t, A_t, Pt, tol=1e-4, iter_max=1000, step_size=0.01, Backtracking_search=False):
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
    alpha = 0.1
    beta = 0.3
    # step_size = 1e-3
    for _ in range(iter_max):
        X_old = X
        fw_old = fw

        grad = B_t @ X - A_t.conj().T
        grad_old = grad

        grad_unit = grad / np.linalg.norm(grad)


        if Backtracking_search:
            counter = 0
            step_size = 1
            while counter <= 50:
                counter += 1
                NablaX = grad_unit
                x_new = X_old - step_size * NablaX
                x_proj = proj(x_new, Pt)
                obj_new = obj_func(x_proj, B_t, A_t)
                obj_benchmark = fw_old - alpha * step_size * np.linalg.norm(grad_unit) ** 2

                if obj_new < obj_benchmark:
                    break
                else:
                    step_size = beta * step_size

        X = X- step_size * grad_unit
        X = proj(X, Pt)
        fw = obj_func(X, B_t, A_t)

        grad_new = B_t @ X - A_t.conj().T
        fw_all.append(fw.item())
        # err1 = np.linalg.norm(grad_new - grad_old) ** 2 / np.linalg.norm(grad_old) ** 2
        err2 = np.abs(fw-fw_old)/np.abs(fw_old)
        if err2 <= tol:
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

    pass
    return Xopt

def Initilize_WnF(H, G, Pt, Nr, Nt, M, C, noise_s):
    M_plus_C = M + C

    # Initialize matrices
    G_s = np.zeros((Nr, Nt, M), dtype=np.complex64)
    G_c = np.zeros((Nr, Nt, C), dtype=np.complex64)

    for iter in range(M_plus_C):
        if iter < M:
            G_s[:, :, iter] = G[(iter * Nr):(iter + 1) * Nr, :]
        else:
            G_c[:, :, iter - M] = G[(iter * Nr):(iter + 1) * Nr, :]


    # Initialize W
    P_k = Pt / K
    norm_H = np.linalg.norm(H, axis=0)
    W = H / norm_H * np.sqrt(P_k)
    Wtemp = W @ W.T.conj()
    W = np.diag(np.sqrt(Pt / Nt / np.diag(Wtemp))) @ W  # Project onto feasible point

    # Initialize F
    F = np.eye(Nr, M)

    # SCNR = np.zeros(M)


    for m in range(M):
        temp = np.eye(Nr, dtype=np.complex64) * noise_s
        for j in range(M_plus_C):
            if j != m:
                G_block = G[(j * Nr):(j + 1) * Nr, :]
                temp += G_block @ Wtemp @ G_block.T.conj()

        A = G[(m * Nr):(m + 1) * Nr, :] @ Wtemp @ G[(m * Nr):(m + 1) * Nr, :].T.conj()

        # Solve generalized eigenvalue problem
        eigvals, eigvecs = scipy.linalg.eig(A, temp)
        # Find the index of the maximum eigenvalue
        index = np.argmax(square_abs(eigvals))
        # print(f'square abs:{square_abs(eigvals)}')
        F[:, m] = np.real(eigvecs[:, index])

        # Compute SCNR
        # SCNR[m] = np.real(F[:, m].T @ A @ F[:, m]) / np.real(F[:, m].T @ temp @ F[:, m])

    return G_s, G_c, W, F


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



def Alg_PGD(H, G, F, W, delta, Pt, mu, NoisePower=1, Iter = 100, tolerance = 1e-4, pgd_lr=1e-2, BTS=False):
    '''
    BTS: Default False. When BTS is False, the fixed step size strategy is used.
    When BTS is True, the dynamic step size strategy is used.
    Iter: the maximum number of iterations.
    tolerance: Tolerance for convergence.
    pgd_lr: step size for fixed step size strategy.
    '''
    Nt, K = H.shape
    Nr, M = F.shape
    M_plus_C  = G.shape[0] // Nr

    G_s = np.zeros((Nr, Nt, M), dtype=np.complex64)
    for iter in range(M):
        G_s[:, :, iter] = G[(iter * Nr):(iter + 1) * Nr, :]


    Obj_cache = []
    SNR_cache = []
    Rate_cache = []

    obj_past = 1

    for i in range(Iter):
        HW_product = H.T.conj() @ W
        # tmp = square_abs(HW_product)
        T_ck = np.sum(square_abs(HW_product), axis=1) + NoisePower
        T_s = calculateT_s(W, G, F, NoisePower)

        # Extract diagonal elements
        # diag_elements = np.diag(HW_product)
        #
        # # Compute the squared absolute values of the diagonal elements
        # diag_squared_abs = square_abs(diag_elements)

        # Compute xi_ck
        xi_ck = T_ck / (T_ck - square_abs(np.diag(HW_product))) - 1
        xi_sm = calculate_scnr(T_s, G_s, W, F)

        z_ck = logsumexp(mu[0], np.log1p(xi_ck))
        z_sm = logsumexp(mu[1], np.log1p(xi_sm))

        theta_ck = np.sqrt(1 + xi_ck) * np.diag(HW_product) / T_ck
        theta_sm = calculate_beta(T_s, G_s, W, xi_sm, F)

        # update F
        F_Last = np.zeros_like(F, dtype=complex)
        # Compute F_Last
        for m in range(M):
            thetam = theta_sm[m, :][np.newaxis, :]
            term = np.sqrt(1 + xi_sm[m]) / (thetam @ thetam.conj().T)
            middle = term * (G_s[:, :, m] @ W @ thetam.conj().T)
            F_Last[:, m] = middle.flatten()

        temp = np.eye(Nr, dtype=complex) * NoisePower
        Wtmp = W @ W.conj().T
        for j in range(M_plus_C):
            G_block = G[j * Nr:(j + 1) * Nr, :]
            temp += G_block @ Wtmp @ G_block.conj().T

        F = np.linalg.inv(temp + Nr * np.eye(Nr)) @ F_Last
        # update W
        # Compute diagonal matrix
        diag_elements = z_ck * np.sqrt(1 + xi_ck) * theta_ck.conj()
        diag_matrix = np.diag(diag_elements)

        # Compute A_c
        A_c = diag_matrix @ np.conj(H.T)
        X = calculateA_s(z_sm, xi_sm, theta_sm, G_s, F)
        B_c = H @ np.diag(z_ck * square_abs(theta_ck)) @ np.conj(H.T)
        Y = calculateB_S(z_sm, theta_sm, G, F)

        A_t = A_c + delta * X
        B_t = B_c + delta * Y

        # W, obj_values = optimize(W, B_t, A_t, Pt, Nt, max_iter=100, tol=1e-4)
        W = PGD_method(W, B_t, A_t, Pt, tol=1e-4, iter_max=1000, step_size=pgd_lr,Backtracking_search=BTS)
        # Plot the objective function values
        # plt.plot(obj_values)
        # plt.xlabel('Iteration')
        # plt.ylabel('Objective Function Value')
        # plt.title('Convergence of the Objective Function')
        # plt.grid(True)
        # plt.show()

        obj = delta * np.min(np.log1p(xi_sm)) + np.min(np.log1p(xi_ck))

        Obj_cache.append(obj)  # Store objective value for each iteration
        SNR_cache.append([np.min(xi_ck), np.min(xi_sm)])  # Store SNR values
        Rate_cache.append([np.min(np.log1p(xi_ck)), np.min(np.log1p(xi_sm))])

        if np.abs(obj - obj_past) / np.abs(obj_past) < tolerance:
            break
        else:
            # Update obj_past and count
            obj_past = obj
    vbest = np.max(Obj_cache)
    index = np.argmax(Obj_cache)

    return Obj_cache, pow2db(SNR_cache[index]), Rate_cache[index], vbest


def Alg_Heuristic(H, G, F, W, delta, Pt, mu, NoisePower=1, Iter = 100, tolerance=1e-4):
    '''
    Iter: the maximum number of iterations.
    tolerance: Tolerance for convergence.
    This algorithm is based on the paper: T. Fang, N. T. Nguyen, and M. Juntti, “Beamforming design for max-min fairness
    performance balancing in ISAC systems,” in Proc. IEEE Works. on Sign. Proc. Adv. in Wirel. Comms., 2024.
    '''
    Nt, K = H.shape
    Nr, M = F.shape
    M_plus_C  = G.shape[0] // Nr

    G_s = np.zeros((Nr, Nt, M), dtype=np.complex64)
    for iter in range(M):
        G_s[:, :, iter] = G[(iter * Nr):(iter + 1) * Nr, :]


    Obj_cache = []
    SNR_cache = []
    Rate_cache = []

    obj_past = 1

    for i in range(Iter):
        HW_product = H.T.conj() @ W
        # tmp = square_abs(HW_product)
        T_ck = np.sum(square_abs(HW_product), axis=1) + NoisePower
        T_s = calculateT_s(W, G, F, NoisePower)

        # Extract diagonal elements
        # diag_elements = np.diag(HW_product)
        #
        # # Compute the squared absolute values of the diagonal elements
        # diag_squared_abs = square_abs(diag_elements)

        # Compute xi_ck
        xi_ck = T_ck / (T_ck - square_abs(np.diag(HW_product))) - 1
        xi_sm = calculate_scnr(T_s, G_s, W, F)

        z_ck = logsumexp(mu[0], np.log1p(xi_ck))
        z_sm = logsumexp(mu[1], np.log1p(xi_sm))

        theta_ck = np.sqrt(1 + xi_ck) * np.diag(HW_product) / T_ck
        theta_sm = calculate_beta(T_s, G_s, W, xi_sm, F)

        # update F
        F_Last = np.zeros_like(F, dtype=complex)
        # Compute F_Last
        for m in range(M):
            thetam = theta_sm[m, :][np.newaxis, :]
            term = np.sqrt(1 + xi_sm[m]) / (thetam @ thetam.conj().T)
            middle = term * (G_s[:, :, m] @ W @ thetam.conj().T)
            F_Last[:, m] = middle.flatten()

        temp = np.eye(Nr, dtype=complex) * NoisePower
        Wtmp = W @ W.conj().T
        for j in range(M_plus_C):
            G_block = G[j * Nr:(j + 1) * Nr, :]
            temp += G_block @ Wtmp @ G_block.conj().T

        F = np.linalg.inv(temp + Nr * np.eye(Nr)) @ F_Last
        # update W
        # Compute diagonal matrix
        diag_elements = z_ck * np.sqrt(1 + xi_ck) * theta_ck.conj()
        diag_matrix = np.diag(diag_elements)

        # Compute A_c
        A_c = diag_matrix @ np.conj(H.T)
        X = calculateA_s(z_sm, xi_sm, theta_sm, G_s, F)
        B_c = H @ np.diag(z_ck * square_abs(theta_ck)) @ np.conj(H.T)
        Y = calculateB_S(z_sm, theta_sm, G, F)

        A_t = A_c + delta * X
        B_t = B_c + delta * Y

        W, obj_values = optimize(W, B_t, A_t, Pt, Nt, max_iter=100, tol=1e-4)
        # W = PGD_method(W, B_t, A_t, Pt, tol=1e-4, iter_max=1000, step_size=pgd_lr,Backtracking_search=BTS)
        # Plot the objective function values
        # plt.plot(obj_values)
        # plt.xlabel('Iteration')
        # plt.ylabel('Objective Function Value')
        # plt.title('Convergence of the Objective Function')
        # plt.grid(True)
        # plt.show()

        obj = delta * np.min(np.log1p(xi_sm)) + np.min(np.log1p(xi_ck))

        Obj_cache.append(obj)  # Store objective value for each iteration
        SNR_cache.append([np.min(xi_ck), np.min(xi_sm)])  # Store SNR values
        Rate_cache.append([np.min(np.log1p(xi_ck)), np.min(np.log1p(xi_sm))])

        if np.abs(obj - obj_past) / np.abs(obj_past) < tolerance:
            break
        else:
            # Update obj_past and count
            obj_past = obj
    vbest = np.max(Obj_cache)
    index = np.argmax(Obj_cache)

    return Obj_cache, pow2db(SNR_cache[index]), Rate_cache[index], vbest

if __name__ == '__main__':

    mu = 10
    Kmax = 12

    Pt = Pt.numpy()
    test_size = 10
    Sum_rate_fs = np.zeros((test_size, 6))
    Sum_rate_ds = np.zeros((test_size, 6))

    SNR_all_fs = np.zeros((test_size, 6, 2))
    SNR_all_ds = np.zeros((test_size, 6, 2))

    Rate_all_fs = np.zeros((test_size, 6, 2))
    Rate_all_ds = np.zeros((test_size, 6, 2))

    data = np.load(data_path_test)
    Hall = data['Hall']
    Gall = data['Gall']

    # Hall = np.zeros((NumCh, Nt, Kmax), dtype=np.complex128)
    # Gall = np.zeros((NumCh, (M+C)*Nr, Nt), dtype=np.complex128)
    for channel_id in range(test_size):
        Hii, Gii = channel(Nt, Nr, Kmax, M, C, seed=channel_id)

        Hall[channel_id, :, :] = Hii
        Gall[channel_id, :, :] = Gii
        for K in range(2, 12+1, 2):
            count = K // 2 - 1
            print(f'Sim.:{channel_id},{K}')
            H = Hii[:,:K]
            G_s, G_c, W, F = Initilize_WnF(H, Gii, Pt, Nr, Nt, M, C, NoisePower)

            obj_iter_fs, snr_iter_fs, rate_iter_fs, rate_fs = Alg_PGD(H, Gii, F, W, delta, Pt, mu=[10., 10.], NoisePower=1, Iter=200,
                                                                      tolerance=1e-4, pgd_lr=1e-2, BTS=False)

            obj_iter_ds, snr_iter_ds, rate_iter_ds, rate_ds = Alg_PGD(H, Gii, F, W, delta, Pt, mu=[10., 10.], NoisePower=1, Iter=200,
                                                                      tolerance=1e-4, pgd_lr=1e-2, BTS=True)
            plt.figure()
            plt.plot(obj_iter_ds)
            plt.xlabel('Iteration')
            plt.ylabel('Objective Function Value')
            plt.title('Convergence of the Objective Function')
            plt.grid(True)
            plt.show()

            # Sum_cache[channel_id, count] = obj_iter[-1]
            # Rate_all[channel_id, count, :] = rate_iter
            Sum_rate_fs[channel_id, count] = rate_fs
            Sum_rate_ds[channel_id, count] = rate_ds
            # Retrieve the SNR_best values from SNR_cache and update SNRopt
            SNR_all_fs[channel_id, count, :] = snr_iter_fs
            SNR_all_ds[channel_id, count, :] = snr_iter_ds

            Rate_all_fs[channel_id, count, :] = rate_iter_fs
            Rate_all_ds[channel_id, count, :] = rate_iter_ds

            pass

    SumRate_avr_fs = np.mean(Sum_rate_fs, axis=0)
    SumRate_avr_ds = np.mean(Sum_rate_ds, axis=0)

    SNR_avr_fs = np.mean(SNR_all_fs, axis=0)
    SNR_avr_ds = np.mean(SNR_all_ds, axis=0)

    Rate_avr_fs = np.mean(Rate_all_fs, axis=0)
    Rate_avr_ds = np.mean(Rate_all_ds, axis=0)


    # data = {
    #     'Hall': Hall,
    #     'Gall': Gall
    # }
    #
    # # Save the data to a .mat file
    # scipy.io.savemat('channel.mat', data)
    # Define x-axis values
    xx = [k for k in range(2, 13, 2)]  # Corresponds to 2:2:12


    plt.figure()
    plt.plot(xx, SumRate_avr_fs, 'r-*', label='Sum min rate (Fixed)')
    plt.plot(xx, SNR_avr_fs[:,0], 'b-s', label='Min commun. rate (Fixed)')
    plt.plot(xx, SNR_avr_fs[:,1], 'k-o', label='Min sensing rate (Fixed)')
    plt.plot(xx, SumRate_avr_ds, 'r--*', label='Sum min rate (Dynamic)')
    plt.plot(xx, SNR_avr_ds[:, 0], 'b--s', label='Min commun. rate (Dynamic)')
    plt.plot(xx, SNR_avr_ds[:, 1], 'k--o', label='Min sensing rate (Dynamic)')
    plt.xlabel('Number of communication users')
    plt.ylabel('Objective')
    plt.legend(prop={'size': 10, 'weight': 'bold'})
    plt.grid(True)
    plt.box(True)
    plt.show()

    # Plot second figure
    plt.figure()
    plt.plot(xx, SNR_avr_fs[:, 0], 'b-s', label='SINR (fixed step size)')
    plt.plot(xx, SNR_avr_fs[:, 1], 'k-o', label='SCNR (fixed step size)')
    plt.plot(xx, SNR_avr_ds[:, 0], 'b--s', label='SINR (dynamic step size)')
    plt.plot(xx, SNR_avr_ds[:, 1], 'k--o', label='SCNR (dynamic step size)')

    plt.xlabel('Number of communication users')
    plt.ylabel('min SINR or min SCNR [dB]')
    plt.legend(prop={'size': 10, 'weight': 'bold'})
    plt.grid(True)
    plt.box(True)
    fig_name = 'TestAlg'
    plt.savefig(directory_model + fig_name + '.png')  # save figure
    plt.show()
    ccc = 1