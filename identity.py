import os
import numpy as np
import glob
import librosa

from speech_tools import *

def main():
    sampling_rate = 22050
    num_mcep = 36
    frame_period = 5.0
    n_frames = 128
    src_speaker = os.listdir('pickles')
    exp_dir = os.path.join('pickles')
    exp_dirs = []
    for f in src_speaker:
        exp_dirs.append(os.path.join(exp_dir, f))

    # load f0 and others
    coded_sps_norms = []
    coded_sps_means = []
    coded_sps_stds = []
    log_f0s_means = []
    log_f0s_stds = []
    for f in exp_dirs:
        coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A = load_pickle(
            os.path.join(f, 'cache{}.p'.format(num_mcep)))
        coded_sps_norms.append(coded_sps_A_norm)
        coded_sps_means.append(coded_sps_A_mean)
        coded_sps_stds.append(coded_sps_A_std)
        log_f0s_means.append(log_f0s_mean_A)
        log_f0s_stds.append(log_f0s_std_A)


    # load wavs and convert
    eval_dirs = os.listdir('datasets_val')
    eval_A_dir = os.path.join('datasets_val', eval_dirs[2])
    print(eval_A_dir)
    for file in glob.glob(eval_A_dir + '/*.wav'):
        alpha = np.random.uniform(0, 1, size=1)
        wav, _ = librosa.load(file, sr=sampling_rate, mono=True)
        wav = wav_padding(wav=wav, sr=sampling_rate, frame_period=frame_period, multiple=4)
        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)

        f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_A, std_log_src=log_f0s_std_A,
                                        mean_log_target=log_f0s_mean_A, std_log_target=log_f0s_std_A)
        coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
        coded_sp_transposed = coded_sp.T
        coded_sp_norm = (coded_sp_transposed - coded_sps_means[2]) / coded_sps_stds[2]
        coded_sp_converted_norm = coded_sp_norm
        #coded_sps_AB_mean = (1-alpha)*coded_sps_means[x_atr]+alpha*coded_sps_means[y_atr]
        #coded_sps_AB_std = (1-alpha)*coded_sps_stds[x_atr]+alpha*coded_sps_stds[y_atr]
        coded_sp_converted = coded_sp_converted_norm * coded_sps_stds[2] + coded_sps_means[2]
        coded_sp_converted = coded_sp_converted.T
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted, fs=sampling_rate)
        wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap, fs=sampling_rate,
                                                 frame_period=frame_period)
        wav_transformed *= 1. / max(0.01, np.max(np.abs(wav_transformed)))
        validation_A_output_dir = 'test'
        os.makedirs(validation_A_output_dir, exist_ok=True)
        librosa.output.write_wav(os.path.join(validation_A_output_dir, 'iden_'+os.path.basename(file)), wav_transformed,
                                 sampling_rate)

if __name__ == '__main__':
    main()
