import os
import glob
import argparse
import tensorflow as tf

from models.cyclegan_vc2 import CycleGAN2
from speech_tools import *

def main(arg):

    source_label = arg.source_label
    target_label = arg.target_label
    alpha = arg.interpolation

    dataset = 'vcc2018'
    src_speakers = os.listdir('pickles')
    model_name = 'cyclegan_vc2_two_step'

    data_dir = os.path.join('datasets', dataset)
    exp_dir = os.path.join('pickles')

    #eval_A_dir = os.path.join(data_dir, 'vcc2018_evaluation', src_speaker)
    source_dir = os.listdir('datasets_val')[source_label]
    eval_A_dir = os.path.join('datasets_val', source_dir)

    exp_A_dir = os.path.join(exp_dir, src_speakers[source_label])

    validation_A_output_dir = os.path.join('experiments', dataset, model_name,
                                           'converted_{}_to_{}_alp{}'.format(
                                           src_speakers[source_label], src_speakers[target_label], alpha))

    os.makedirs(validation_A_output_dir, exist_ok=True)

    sampling_rate = 22050
    num_mcep = 36
    frame_period = 5.0
    n_frames = 128

    print('Loading cached data...')
    exp_dirs = []
    for f in src_speakers:
        exp_dirs.append(os.path.join(exp_dir, f))

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

    num_domains = len(coded_sps_norms)
    model = CycleGAN2(num_features=num_mcep, num_domains=num_domains, batch_size=1, mode='test')
    ckpt = tf.train.get_checkpoint_state(os.path.join('experiments', model_name, 'checkpoints'))
    if ckpt:
        #last_model = ckpt.all_model_checkpoint_paths[1]
        last_model = ckpt.model_checkpoint_path
        print("loading {}".format(last_model))
        model.load(filepath=last_model)
    else:
        print("checkpoints are not found")

    print('Generating Validation Data ...')
    for file in glob.glob(eval_A_dir + '/*.wav'):
        wav, _ = librosa.load(file, sr=sampling_rate, mono=True)
        wav = wav_padding(wav=wav, sr=sampling_rate, frame_period=frame_period, multiple=4)
        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
        log_f0s_mean_AB = (1-alpha)*log_f0s_means[source_label]+alpha*log_f0s_means[target_label]
        log_f0s_std_AB = (1-alpha)*log_f0s_stds[source_label]+alpha*log_f0s_stds[target_label]
        f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_means[source_label], std_log_src=log_f0s_stds[source_label],
                                        mean_log_target=log_f0s_mean_AB, std_log_target=log_f0s_std_AB)
        coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
        coded_sp_transposed = coded_sp.T
        coded_sp_norm = (coded_sp_transposed - coded_sps_means[source_label]) / coded_sps_stds[source_label]
        x_label = source_label
        y_label = target_label
        x_labels = np.zeros([1, num_domains])
        y_labels = np.zeros([1, num_domains])
        for b in range(1):
            x_labels[b] = np.identity(num_domains)[x_label]
            y_labels[b] = np.identity(num_domains)[y_label]

        alpha_t = alpha*np.ones([1])
        coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm]), label_A=x_labels, label_B=y_labels, alpha=alpha_t)[0]
        if coded_sp_converted_norm.shape[1] > len(f0):
            coded_sp_converted_norm = coded_sp_converted_norm[:, :-1]
        coded_sps_AB_mean = (1-alpha)*coded_sps_means[source_label]+alpha*coded_sps_means[target_label]
        coded_sps_AB_std = (1-alpha)*coded_sps_stds[source_label]+alpha*coded_sps_stds[target_label]
        coded_sp_converted = coded_sp_converted_norm * coded_sps_AB_std + coded_sps_AB_mean
        coded_sp_converted = coded_sp_converted.T
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted, fs=sampling_rate)
        wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap, fs=sampling_rate,
                                                 frame_period=frame_period)
        wav_transformed *= 1. / max(0.01, np.max(np.abs(wav_transformed)))
        librosa.output.write_wav(os.path.join(validation_A_output_dir, os.path.basename(file)), wav_transformed,
                                 sampling_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_label',"-s", dest='source_label', type=int, default=None, help='source label')
    parser.add_argument('--target_label',"-t", dest='target_label', type=int, default=None, help='target label')
    parser.add_argument('--interpolation',"-ip", dest='interpolation', type=float, default=1.0, help='interpolation late(0<=interp<=1)')
    args = parser.parse_args()

    main(args)
