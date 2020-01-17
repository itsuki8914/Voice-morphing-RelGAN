import os
import time

from speech_tools import *
import numpy as np
from multiprocessing import Pool

datasets_dir = "datasets_splitted/datasets"

def process(folder):
    dataset = folder

    data_dir = datasets_dir
    exp_dir = os.path.join('pickles', dataset)

    train_A_dir = os.path.join(data_dir, folder)

    exp_A_dir = exp_dir

    os.makedirs(exp_A_dir, exist_ok=True)

    sampling_rate = 22050
    num_mcep = 36
    frame_period = 5.0
    n_frames = 128

    print('Loading Wavs...')

    start_time = time.time()

    wavs_A = load_wavs(wav_dir=train_A_dir, sr=sampling_rate)

    print('Extracting acoustic features...')

    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(wavs=wavs_A, fs=sampling_rate,
                                                                     frame_period=frame_period, coded_dim=num_mcep)

    print('Calculating F0 statistics...')

    log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)

    print('Log Pitch A')
    print('Mean: %f, Std: %f' % (log_f0s_mean_A, log_f0s_std_A))

    print('Normalizing data...')

    coded_sps_A_transposed = transpose_in_list(lst=coded_sps_A)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(
        coded_sps=coded_sps_A_transposed)

    print(np.max(coded_sps_A_norm[0]))
    print(np.min(coded_sps_A_norm[0]))
    print('Saving data...')
    save_pickle(os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep)),
                (coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A))


    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Preprocessing Done.')

    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (
        time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

if __name__ == '__main__':
    folders = os.listdir(datasets_dir)
    TIME= time.time()
    cores = min(len(folders), 4)
    parallel = True
    if parallel:
        p = Pool(cores)
        p.map(process, folders)

        p.close()
    else:
        for f in folders:
            process(f)

    print(time.time()-TIME)
