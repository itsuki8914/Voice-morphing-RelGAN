import librosa
import numpy as np
import pyworld
import glob
import pickle
from tqdm import tqdm


def load_wavs(wav_dir, sr):
    wavs = list()
    for file in glob.glob(wav_dir + '/*.wav'):
        wav, _ = librosa.load(file, sr=sr, mono=True)
        #wav *= 1. / max(0.01, np.max(np.abs(wav)))
        wavs.append(wav)

    return wavs


def world_decompose(wav, fs, frame_period=5.0):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0)
    #print("f0 ",f0)
    spectrogram = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    aperiodicity = pyworld.d4c(wav, f0, timeaxis, fs)
    #print("ap ",ap)

    return f0, timeaxis, spectrogram, aperiodicity


def world_encode_spectral_envelop(sp, fs, dim=24):
    # Get Mel-cepstral coefficients (MCEPs)

    # sp = sp.astype(np.float64)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)

    return coded_sp


def world_decode_spectral_envelop(coded_sp, fs):
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    # coded_sp = coded_sp.astype(np.float32)
    # coded_sp = np.ascontiguousarray(coded_sp)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)

    return decoded_sp


def world_encode_data(wavs, fs, frame_period=5.0, coded_dim=24):
    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    coded_sps = list()

    for wav in tqdm(wavs):
        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=fs, frame_period=frame_period)
        coded_sp = world_encode_spectral_envelop(sp=sp, fs=fs, dim=coded_dim)
        if coded_sp.shape[0]<128: continue
        f0s.append(f0)
        timeaxes.append(timeaxis)
        sps.append(sp)
        aps.append(ap)
        coded_sps.append(coded_sp)

    return f0s, timeaxes, sps, aps, coded_sps


def transpose_in_list(lst):
    transposed_lst = list()
    for array in lst:
        transposed_lst.append(array.T)
    return transposed_lst


def world_decode_data(coded_sps, fs):
    decoded_sps = list()

    for coded_sp in coded_sps:
        decoded_sp = world_decode_spectral_envelop(coded_sp, fs)
        decoded_sps.append(decoded_sp)

    return decoded_sps


def world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period):
    # decoded_sp = decoded_sp.astype(np.float64)
    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    # Librosa could not save wav if not doing so
    wav = wav.astype(np.float32)

    return wav


def world_synthesis_data(f0s, decoded_sps, aps, fs, frame_period):
    wavs = list()

    for f0, decoded_sp, ap in zip(f0s, decoded_sps, aps):
        wav = world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period)
        wavs.append(wav)

    return wavs


def coded_sps_normalization_fit_transoform(coded_sps):
    coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=True)

    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)

    return coded_sps_normalized, coded_sps_mean, coded_sps_std


def coded_sps_normalization_transoform(coded_sps, coded_sps_mean, coded_sps_std):
    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)

    return coded_sps_normalized


def coded_sps_normalization_inverse_transoform(normalized_coded_sps, coded_sps_mean, coded_sps_std):
    coded_sps = list()
    for normalized_coded_sp in normalized_coded_sps:
        coded_sps.append(normalized_coded_sp * coded_sps_std + coded_sps_mean)

    return coded_sps


def coded_sp_padding(coded_sp, multiple=4):
    num_features = coded_sp.shape[0]
    num_frames = coded_sp.shape[1]
    num_frames_padded = int(np.ceil(num_frames / multiple)) * multiple
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    coded_sp_padded = np.pad(coded_sp, ((0, 0), (num_pad_left, num_pad_right)), 'constant', constant_values=0)

    return coded_sp_padded


def wav_padding(wav, sr, frame_period, multiple=4):
    assert wav.ndim == 1
    num_frames = len(wav)
    num_frames_padded = int(
        (np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (
                sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values=0)

    return wav_padded


def logf0_statistics(f0s):
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std


def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target, eps=1e-8):
    # Logarithm Gaussian normalization for Pitch Conversions
    f0_converted = np.exp((np.log(f0+eps) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_converted


def wavs_to_specs(wavs, n_fft=1024, hop_length=None):
    stfts = list()
    for wav in wavs:
        stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
        stfts.append(stft)

    return stfts


def wavs_to_mfccs(wavs, sr, n_fft=1024, hop_length=None, n_mels=128, n_mfcc=24):
    mfccs = list()
    for wav in wavs:
        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)
        mfccs.append(mfcc)

    return mfccs


def mfccs_normalization(mfccs):
    mfccs_concatenated = np.concatenate(mfccs, axis=1)
    mfccs_mean = np.mean(mfccs_concatenated, axis=1, keepdims=True)
    mfccs_std = np.std(mfccs_concatenated, axis=1, keepdims=True)

    mfccs_normalized = list()
    for mfcc in mfccs:
        mfccs_normalized.append((mfcc - mfccs_mean) / mfccs_std)

    return mfccs_normalized, mfccs_mean, mfccs_std


def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def sample_train_data(dataset_A, nBatch, num_mcep=36, n_frames=128):
    x = np.zeros((nBatch, num_mcep, n_frames), dtype=np.float32)
    x2 = np.zeros((nBatch, num_mcep, n_frames), dtype=np.float32)
    y = np.zeros((nBatch, num_mcep, n_frames), dtype=np.float32)
    z = np.zeros((nBatch, num_mcep, n_frames), dtype=np.float32)
    x_atr = []
    y_atr = []
    z_atr = []
    for i in range(nBatch):
        labels = np.arange(len(dataset_A))
        atr = np.random.choice(labels)
        x_idx = np.random.choice(len(dataset_A[atr]))
        data_x = dataset_A[atr][x_idx]
        frames_x_total = data_x.shape[1]
        assert frames_x_total >= n_frames
        start_x = np.random.randint(frames_x_total - n_frames + 1)
        end_x = start_x + n_frames
        x[i,:,:] = data_x[:,start_x:end_x]
        if np.random.random() >0.9995:
            x[i] *= 0
        x_atr.append(atr)

        x2_idx = np.random.choice(len(dataset_A[atr]))
        data_x2 = dataset_A[atr][x2_idx]
        frames_x2_total = data_x2.shape[1]
        assert frames_x2_total >= n_frames
        start_x2 = np.random.randint(frames_x2_total - n_frames + 1)
        end_x2 = start_x2 + n_frames
        x2[i,:,:] = data_x2[:,start_x2:end_x2]
        if np.random.random() >0.9995:
            x2[i] *= 0

        labels = labels[labels!=atr]

        atr = np.random.choice(labels)
        y_idx = np.random.choice(len(dataset_A[atr]))
        data_y = dataset_A[atr][y_idx]
        frames_y_total = data_y.shape[1]
        assert frames_y_total >= n_frames
        start_y = np.random.randint(frames_y_total - n_frames + 1)
        end_y = start_y + n_frames
        y[i,:,:] = data_y[:,start_y:end_y]
        if np.random.random() >0.9995:
            y[i] *= 0
        y_atr.append(atr)
        labels = labels[labels!=atr]

        atr = np.random.choice(labels)
        z_idx = np.random.choice(len(dataset_A[atr]))
        data_z = dataset_A[atr][z_idx]
        frames_z_total = data_z.shape[1]
        assert frames_z_total >= n_frames
        start_z = np.random.randint(frames_z_total - n_frames + 1)
        end_z = start_z + n_frames
        z[i,:,:] = data_z[:,start_z:end_z]
        if np.random.random() >0.9995:
            z[i] *= 0
        z_atr.append(atr)

    return x, x2, x_atr, y, y_atr, z, z_atr
