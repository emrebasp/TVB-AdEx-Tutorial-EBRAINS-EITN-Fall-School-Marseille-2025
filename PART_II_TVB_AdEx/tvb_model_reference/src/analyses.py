import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.signal as signal
import matplotlib.pyplot as plt

from numba import njit
from matplotlib.pyplot import cm
from scipy.integrate import simps
from scipy.optimize import curve_fit
from matplotlib.tri import Triangulation
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import butter, sosfilt, sosfreqz


def mean_FC(FR, do_plot=False):
    """Returns the average value of the FC matrix obtained from Pearson correlation of timetraces in FR."""
    FC = np.corrcoef(np.transpose(FR))
    FC_mean = np.mean(FC) - np.trace(FC) / FC.size  # Delete the relevance of the diagonal elements.
    if do_plot:
        im = plt.imshow(FC)
        plt.title('$FC$ matrix')
        plt.xlabel('Nodes')
        plt.ylabel('Nodes')
        plt.colorbar(im)
        plt.tight_layout()
        plt.show()
        plt.close()
    return FC_mean


def mean_PLI(FR, do_plot=False):
    nnodes = FR.shape[1]
    sig = np.transpose(FR)
    PLI = np.zeros((nnodes, nnodes))
    hilb_amplitude_region = np.zeros_like(sig)
    hilb_phase_region = np.zeros_like(sig)

    for reg in range(nnodes):
        hilb = signal.hilbert(sig[reg])
        hilb_amplitude_region[reg] = np.abs(hilb)
        hilb_phase_region[reg] = np.angle(hilb)

    for i_reg in range(nnodes):
        for j_reg in range(i_reg, nnodes):
            phase_lags = hilb_phase_region[i_reg] - hilb_phase_region[j_reg]
            PLI[i_reg][j_reg] = np.abs(np.mean(np.sign(phase_lags)))
            PLI[j_reg][i_reg] = PLI[i_reg][j_reg]

    PLI_mean = np.mean(PLI) - np.trace(PLI) / PLI.size  # Not interested about the contribution of diagonal elements
    if do_plot:
        im = plt.imshow(PLI)
        plt.title('$PLI$ matrix')
        plt.xlabel('Nodes')
        plt.ylabel('Nodes')
        plt.colorbar(im)
        plt.tight_layout()
        plt.show()
        plt.close()
    return PLI_mean


def mean_UD_duration(FR, dt, ratio_threshold=0.3,
                     len_state=20,
                     gauss_width_ratio=10,
                     units='s'):
    N, M = FR.shape
    sampling_rate = 1 / dt
    up_mean_len = 0
    down_mean_len = 0
    up_total_len = 0
    down_total_len = 0
    for m in range(M):  # Sweep over regions
        _, _, train_bool, _ = detect_UP(FR[:, m], ratioThreshold=ratio_threshold,
                                        sampling_rate=sampling_rate,
                                        len_state=len_state,
                                        gauss_width_ratio=gauss_width_ratio)

        mean_up_m, mean_down_m, total_up, total_down = obtain_updown_durs(train_bool, dt)
        # We update the running value that will be transformed in mean of means at the end
        up_mean_len += mean_up_m
        down_mean_len += mean_down_m
        up_total_len += total_up
        down_total_len += total_down

    if units == 'ms':
        up_mean_len = up_mean_len / M
        down_mean_len = down_mean_len / M
        up_total_len = up_total_len / M
        down_total_len = down_total_len / M
    elif units == 's':
        up_mean_len = up_mean_len / (1000 * M)  # Since dt is in ms
        down_mean_len = down_mean_len / (1000 * M)
        up_total_len = up_total_len / (1000 * M)
        down_total_len = down_total_len / (1000 * M)
    else:
        raise ValueError('Choose units between s or ms')
    return up_mean_len, down_mean_len, up_total_len, down_total_len


def psd_fmax_ampmax(frq, psd, type_mean='avg_psd', prominence=1, which_peak='both'):
    """ Returns frequency at which PSD peaks and the amplitude of the peak. Makes use of scipy.signal.find_peaks.

    Parameters
    ----------
    frq: array_like
        Vector of frequencies, array of shape (N, )

    psd: array_like
        Array containing the PSD (or PSDs). If single PSD, array of shape (N, ). If multiple (M)
        PSDs, array of shape (M, N). So, in each row a different PSD

    type_mean: str
        For multiple PSDs.
        'avg_psd': For looking for the max of the average of all the PSDs
        'avg_slopes': For looking for the f_max_i and amp_max_i of each of the M PSD and obtaining mean(f_max_i) and
        mean(amp_max_i).

    prominence: Understand how prominences work and note it down here. Give a valuable range as well

    which_peak: str
        To choose which peak we want the function to return:
        - 'max_prominence': Returns the peak with the maximum prominence
        - 'max_amplitude': Returns the peak with the maximum power value
        - 'both': Returns both peaks in a list

    Returns
    -------
    f_max: list
        Frequency at which we find the peak of the PSD. 2 elements if 'both'
    amp_max: list
        Amplitude of the chosen peak of the PSD. 2 elements if 'both'
    """
    good_idxs = frq > 0
    frq = frq[good_idxs]

    if not len(psd.shape) == 1 and type_mean == 'avg_psd':  # 2D PSDs and avg_psd, obtain the average of PSD
        psd = np.mean(psd, axis=0)
        psd = psd[good_idxs]

    if len(psd.shape) == 1 or type_mean == 'avg_psd':  # Now PSD is only a vector
        idx_peaks, props = signal.find_peaks(psd, prominence=prominence)
        # TODO: Find how to obtain a reliable prominence for our simulations
        if idx_peaks.size == 0:  # It might be that the peak finding algorithm does not find any
            if which_peak == 'max_amplitude' or which_peak == 'max_prominence':
                return [np.nan], [np.nan]
            elif which_peak == 'both':
                return [np.nan, np.nan], [np.nan, np.nan]
            else:
                raise ValueError('Choose a correct value of which_peak')
        else:
            # We take the peak with maximum prominence:
            idx_max_peak_prom = idx_peaks[np.argmax(props['prominences'])]
            amp_max_prom = psd[idx_max_peak_prom]

            # Or we take the peak that has the highest PSD value:
            idx_max_peak = idx_peaks[np.argmax(psd[idx_peaks])]
            amp_max = psd[idx_max_peak]

            if which_peak == 'max_prominence':
                return [frq[idx_max_peak_prom]], [amp_max_prom]
            elif which_peak == 'max_amplitude':
                return [frq[idx_max_peak]], [amp_max]
            elif which_peak == 'both':
                return [frq[idx_max_peak], frq[idx_max_peak_prom]], [amp_max, amp_max_prom]
            else:
                raise ValueError('Choose a correct value of which_peak')

    elif not len(psd.shape) == 1 and type_mean == 'avg_slopes':  # We want PSD to be 2D array to obtain means of each
        f_max_m = np.array([])
        amp_max_m = np.array([])
        f_max_m_prom = np.array([])
        amp_max_m_prom = np.array([])

        psd = psd[:, good_idxs]
        M, N = psd.shape
        flag = False
        for m in range(M):
            aux_psd = psd[m]
            idx_peaks, props = signal.find_peaks(aux_psd, prominence=1)

            if idx_peaks.size == 0:  # Avoid errors if the algorithm cannot find any peaks
                flag = True
                break

            idx_max_peak = idx_peaks[np.argmax(psd[idx_peaks])]
            idx_max_peak_prom = idx_peaks[np.argmax(props['prominences'])]

            f_max_m = np.append(f_max_m, frq[idx_max_peak])
            f_max_m_prom = np.append(f_max_m_prom, frq[idx_max_peak_prom])

            amp_max_m = np.append(amp_max_m, psd[idx_max_peak])
            amp_max_m_prom = np.append(amp_max_m_prom, psd[idx_max_peak_prom])

        if flag:
            return [np.nan], [np.nan]
        else:
            if which_peak == 'max_amplitude':
                return [np.mean(f_max_m)], [np.mean(amp_max_m)]
            elif which_peak == 'max_prominence':
                return [np.mean(f_max_m_prom)], [np.mean(amp_max_m_prom)]
            elif which_peak == 'both':
                return [np.mean(f_max_m), np.mean(f_max_m_prom)], [np.mean(amp_max_m), np.mean(amp_max_m_prom)]
            else:
                raise ValueError('Choose a correct value of which_peak')

    else:
        print('Check size of array if compatible with type of average. No computation done.')
        return [np.nan], [np.nan]


def detect_UP(train_cut, ratioThreshold=0.4,
              sampling_rate=1., len_state=50.,
              gauss_width_ratio=10., min_for_up=0.2):
    """
    detect UP states from time signal
    (population rate or population spikes or cell voltage trace)
    return start and ends of states.

    Written by Trang-Anh Nghiem. Modified with min_for_up by David Aquilue

    Parameters
    ----------
    train_cut: array
        array of shape (N, ) containing the time trace on which we detect upstates

    ratioThreshold: float
        Over which % of the FR maximum value in the time trace of a region we consider an up-state

    sampling_rate: float
        Sampling rate of the time trace. Usually 1 / dt. In ms**(-1)

    len_state: float
        Minimum length (in ms) of time over threshold to be considered up-state (I think)

    gauss_width_ratio: float
        Width ratio of the Gaussian Kernel used in the filter for detecting up-states.

    min_for_up: float
        A value under which there is no up-state. That way, if we have high relative variations
        near 0 value but the FR is not higher than 0.3 there will be no up-state.
        However, take into account that this will modify the functioning of the algorithm, possibly
        underestimating up state duration.

    Returns
    -------
    idx: array
        indexes where there is a change of state.
    train_shift: array
        time trace of the filtered signal - ratioThreshold * np.max(train_filtered)
    train_bool: array
        array containing 1s when up state and 0s when downstate
    """
    # convolve with Gaussian
    time = range(len(train_cut))  # indexes
    gauss_width = gauss_width_ratio * sampling_rate

    # We obtain a gauss filter
    gauss_filter = np.exp(-0.5 * ((np.subtract(time, len(train_cut) / 2.0) / gauss_width) ** 2))
    gauss_norm = np.sqrt(2 * np.pi * gauss_width ** 2)
    gauss_filter = gauss_filter / gauss_norm

    # We filter the signal by convolving the gauss_filter
    train_filtered = signal.fftconvolve(train_cut, gauss_filter)
    train_filt = train_filtered[int(len(train_cut) / 2.0): \
                                int(3 * len(train_cut) / 2.0)]
    thresh = ratioThreshold * np.max(train_filt)

    # times at which filtered signal crosses threshold
    train_shift = np.subtract(train_filt, thresh) - min_for_up
    idx = np.where(np.multiply(train_shift[1:], train_shift[:-1]) < 0)[0]

    # train of 0 in DOWN state and 1 in UP state
    train_bool = np.zeros(len(train_shift))
    train_bool[train_shift > 0] = 1  # assign to 1 in UP states

    # cut states shorter than min length
    idx = np.concatenate(([0], idx, [len(train_filt)]))
    diff_remove = np.where(np.diff(idx) < len_state * sampling_rate)[0]
    idx_start_remove = idx[diff_remove]
    idx_end_remove = idx[np.add(diff_remove, 1)] + 1

    for ii_start, ii_end in zip(idx_start_remove, idx_end_remove):
        train_bool[ii_start:ii_end] = np.ones_like(train_bool[ii_start:ii_end]) * train_bool[ii_start - 1]
        # assign to same state as previous long

    idx = np.where(np.diff(train_bool) != 0)[0]
    idx = np.concatenate(([0], idx, [len(train_filt)])) / sampling_rate
    return idx, train_shift, train_bool, thresh


def obtain_updown_durs(train_bool, dt):
    N = train_bool.size
    up_durs = np.empty(0)  # Something of the like up_durs = []
    down_durs = np.empty(0)
    current_up_duration = 0
    current_down_duration = 0
    for i in range(1, N):  # We sweep over all the values of the train_bool signal
        if train_bool[i - 1] == train_bool[i]:  # If 2 consecutive equal values -> increase state duration
            if train_bool[i - 1] == 1:
                current_up_duration += dt
            else:
                current_down_duration += dt
        else:  # If 2 consecutive NOT equal values -> increase state duration + store duration + restore
            if train_bool[i - 1] == 1:
                up_durs = np.append(up_durs, current_up_duration)
                current_up_duration = 0
            else:
                down_durs = np.append(down_durs, current_down_duration)
                current_down_duration = 0
        if i == N - 1:  # Regardless of the value of the last time point, we have to store the last duration.
            if train_bool[i] == 1:
                current_up_duration += dt
                up_durs = np.append(up_durs, current_up_duration)
                current_up_duration = 0
            else:
                current_down_duration += dt
                down_durs = np.append(down_durs, current_down_duration)
                current_down_duration = 0

    if up_durs.size == 0:  # If no up-states, return duration of 0
        mean_up_durs = 0
        total_up_durs = 0
    else:
        mean_up_durs = np.mean(up_durs)
        total_up_durs = np.sum(up_durs)

    if down_durs.size == 0:  # If no down-states, return duration of 0
        mean_down_durs = 0
        total_down_durs = 0
    else:
        mean_down_durs = np.mean(down_durs)
        total_down_durs = np.sum(down_durs)

    return mean_up_durs, mean_down_durs, total_up_durs, total_down_durs


# filter in delta

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def mean_PLI_filt(FR, dt=0.1e-3, lowcut=0.5, highcut=4, order=2, threshold=0.9, do_plot=False):
    fs = 1 / dt
    nyq = 0.5 * fs  # Nyquist Frequency
    nnodes = FR.shape[1]

    y = []
    for i in range(nnodes):
        y1 = butter_bandpass_filter(FR[:, i], lowcut, highcut, fs, order=order)
        y.append(y1)

    y = np.array(y).T

    sig = np.transpose(y)
    PLI = np.zeros((nnodes, nnodes))
    hilb_phase_region = np.zeros_like(sig)

    for reg in range(nnodes):
        hilb = signal.hilbert(sig[reg])
        hilb_phase_region[reg] = np.angle(hilb)

    for i_reg in range(nnodes):
        for j_reg in range(i_reg, nnodes):
            phase_lags = hilb_phase_region[i_reg] - hilb_phase_region[j_reg]
            filtered_phase_lag = phase_lags[np.abs(phase_lags) > threshold]

            PLI[i_reg][j_reg] = np.abs(np.mean(np.sign(filtered_phase_lag)))
            PLI[j_reg][i_reg] = PLI[i_reg][j_reg]

    PLI_mean = np.nanmean(
        PLI)  # - np.trace(PLI) / PLI.size  # Not interested about the contribution of diagonal elements
    if do_plot:
        im = plt.imshow(PLI)
        plt.title('$PLI$ matrix')
        plt.xlabel('Nodes')
        plt.ylabel('Nodes')
        plt.colorbar(im)
        plt.tight_layout()
        plt.show()
        plt.close()
    return PLI_mean
