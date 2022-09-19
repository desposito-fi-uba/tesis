import numpy as np


def lms(u, d, M, mu=0.01):
    d = d.flatten()
    u = u.flatten()
    N = len(u)
    w = np.zeros((M,))
    padded_u = np.zeros((M-1 + N,))
    padded_u[M-1:] = u
    u = padded_u
    y = np.zeros((N,))
    e = np.zeros((N,))
    w_n = np.zeros((M, N))
    for i in range(0, N):
        U = u[i:i + M]
        U = U[::-1]
        y[i] = np.inner(U, w)
        e[i] = d[i] - y[i]
        w_n[:, i] = w
        w = w + mu * U * e[i]

    return y, e, w_n


def rls(u, d, M, lmda=0.998, epsilon=0.01, weights_seed=None, update_matrix_seed=None):
    d = d.flatten()
    u = u.flatten()
    N = len(u)
    padded_u = np.zeros((M-1 + N, 1))
    padded_u[M-1:, 0] = u
    u = padded_u
    P = (1/epsilon) * np.eye(M) if update_matrix_seed is None else update_matrix_seed
    w = np.zeros((M, 1)) if weights_seed is None else weights_seed
    y = np.zeros((N, 1))
    e = np.zeros((N, 1))
    w_n = np.zeros((M, N))
    lmda_inv = 1/lmda
    for i in range(0, N):
        U = u[i:i + M]
        U = U[::-1, :]
        P_update_num = lmda_inv * np.dot(P, np.dot(U, np.dot(np.transpose(U), P)))
        P_update_den = 1 + lmda_inv * np.dot(np.transpose(U), np.dot(P, U))
        P_update = P_update_num / P_update_den
        P = lmda_inv * (P - P_update)
        w_n[:, i] = w[:, 0]
        w = w + np.dot(P, U) * (d[i] - np.dot(np.transpose(U), w))
        y[i] = np.dot(np.transpose(U), w)
        e[i] = d[i] - y[i]

    return y.flatten(), e.flatten(), w_n, w, P


def filter_signal(signal, filter_size, weights):
    u = signal.flatten()
    N = len(u)
    M = filter_size
    padded_u = np.zeros((M-1 + N, 1))
    padded_u[M-1:, 0] = u
    u = padded_u
    y = np.zeros((N, 1))
    for i in range(0, N):
        U = u[i:i + M]
        U = U[::-1, :]
        y[i] = np.dot(np.transpose(U), weights)

    return y.flatten()


def denoise_signal(noisy_speech, correlated_noise, sample_rate, filter_size=100, forgetting_rate=0.99, time_window=20):
    assert len(noisy_speech) == len(correlated_noise)
    amount_samples = len(noisy_speech)
    filter_size = filter_size
    forgetting_rate = forgetting_rate
    time_window = time_window / 1000
    frame_size = int(time_window * sample_rate)
    amount_chunks = amount_samples // frame_size

    cleaned_speech = np.zeros((amount_samples,))
    update_matrix_seed = None
    weights_seed = None
    for i in range(amount_chunks):
        noisy_speech_frame = noisy_speech[i * frame_size:(i + 1) * frame_size]
        correlated_noise_frame = correlated_noise[i * frame_size:(i + 1) * frame_size]

        _, cleaned_speech_frame, weights, weights_seed, update_matrix_seed = rls(
            correlated_noise_frame, noisy_speech_frame, filter_size, lmda=forgetting_rate, weights_seed=weights_seed,
            update_matrix_seed=update_matrix_seed
        )
        cleaned_speech[i * frame_size:(i + 1) * frame_size] = cleaned_speech_frame

    return cleaned_speech
