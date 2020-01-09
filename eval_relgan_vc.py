import os
import glob
import argparse
import tensorflow as tf

from models.relgan import RelGAN
from speech_tools import *

def main(arg):

    source_label = arg.source_label
    target_label = arg.target_label
    x_atr = arg.source_label
    y_atr = arg.target_label
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
    model = RelGAN(num_features=num_mcep, num_domains=num_domains, batch_size=1, mode='test')
    ckpt = tf.train.get_checkpoint_state(os.path.join('experiments', model_name, 'checkpoints'))
    if ckpt:
        #last_model = ckpt.all_model_checkpoint_paths[1]
        last_model = ckpt.model_checkpoint_path
        print("loading {}".format(last_model))
        model.load(filepath=last_model)
    else:
        print("checkpoints are not found")
        print("to inference must need a trained model.")
        return

    print('Generating Validation Data ...')
    for file in glob.glob(eval_A_dir + '/*.wav'):
        eval_dirs = os.listdir('datasets_val')
        assert len(eval_dirs) == num_domains
        x,x2, x_atr, y, y_atr, z, z_atr = sample_train_data(dataset_A=coded_sps_norms, nBatch=1, num_mcep=num_mcep, n_frames=n_frames)
        x_labels = np.zeros([1, num_domains])
        y_labels = np.zeros([1, num_domains])
        for b in range(1):
            x_labels[b] = np.identity(num_domains)[x_atr[b]]
            y_labels[b] = np.identity(num_domains)[y_atr[b]]
        x_atr =x_atr[0]
        y_atr =y_atr[0]
        eval_A_dir = os.path.join('datasets_val', eval_dirs[x_atr])
        print(eval_A_dir)
        for file in glob.glob(eval_A_dir + '/*.wav'):
            for i in range(1,4):
                alpha =np.ones(1)*i/3
                #alpha = np.random.uniform(0, 1, size=1)
                wav, _ = librosa.load(file, sr=sampling_rate, mono=True)
                wav *= 1. / max(0.01, np.max(np.abs(wav)))
                wav = wav_padding(wav=wav, sr=sampling_rate, frame_period=frame_period, multiple=4)
                f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
                f0s_mean_A = np.exp(log_f0s_means[x_atr])
                f0s_mean_B = np.exp(log_f0s_means[y_atr])
                f0s_mean_AB = alpha * f0s_mean_B + (1-alpha) * f0s_mean_A
                log_f0s_mean_AB = np.log(f0s_mean_AB)
                f0s_std_A = np.exp(log_f0s_stds[x_atr])
                f0s_std_B = np.exp(log_f0s_stds[y_atr])
                f0s_std_AB = alpha * f0s_std_B + (1-alpha) * f0s_std_A
                log_f0s_std_AB = np.log(f0s_std_AB)
                f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_means[x_atr], std_log_src=log_f0s_stds[x_atr],
                                                mean_log_target=log_f0s_mean_AB, std_log_target=log_f0s_std_AB)
                coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
                coded_sp_transposed = coded_sp.T
                coded_sp_norm = (coded_sp_transposed - coded_sps_means[x_atr]) / coded_sps_stds[x_atr]
                coded_sp_norm = np.tanh(coded_sp_norm)
                coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm]), label_A=x_labels, label_B=y_labels, alpha=alpha)[0]
                coded_sp_converted_norm = np.arctanh(coded_sp_converted_norm)
                #coded_sp_converted_norm = coded_sp_norm
                if coded_sp_converted_norm.shape[1] > len(f0):
                    coded_sp_converted_norm = coded_sp_converted_norm[:, :-1]
                coded_sps_AB_mean = (1-alpha)*coded_sps_means[x_atr]+alpha*coded_sps_means[y_atr]
                coded_sps_AB_std = (1-alpha)*coded_sps_stds[x_atr]+alpha*coded_sps_stds[y_atr]
                coded_sp_converted = coded_sp_converted_norm * coded_sps_AB_std + coded_sps_AB_mean
                coded_sp_converted = coded_sp_converted.T
                coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted, fs=sampling_rate)

                wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap, fs=sampling_rate,
                                                         frame_period=frame_period)

                wav_transformed *= 1. / max(0.01, np.max(np.abs(wav_transformed)))
                validation_A_output_dir = 'test2'
                os.makedirs(validation_A_output_dir, exist_ok=True)
                librosa.output.write_wav(os.path.join(validation_A_output_dir, "{}_to_{}_{:.3f}_{}".format(x_atr,y_atr,alpha[0],os.path.basename(file))), wav_transformed,
                                         sampling_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_label',"-s", dest='source_label', type=int, default=None, help='source label')
    parser.add_argument('--target_label',"-t", dest='target_label', type=int, default=None, help='target label')
    parser.add_argument('--interpolation',"-ip", dest='interpolation', type=float, default=1.0, help='interpolation late(0<=interp<=1)')
    args = parser.parse_args()

    main(args)
