import os
import random
import numpy as np
import glob
import librosa
import tensorflow as tf
from models.relgan import RelGAN
from speech_tools import load_pickle, sample_train_data
from speech_tools import *

seed = 65535

random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

def main():
    dataset = 'datasets'
    src_speaker = os.listdir('pickles')
    model_name = 'relgan_vm'
    os.makedirs(os.path.join('experiments', model_name, 'checkpoints'), exist_ok=True)
    log_dir = os.path.join('logs', model_name)
    os.makedirs(log_dir, exist_ok=True)


    exp_dir = os.path.join('pickles')

    exp_dirs = []
    for f in src_speaker:
        exp_dirs.append(os.path.join(exp_dir, f))
    print(exp_dirs)

    # Data parameters
    sampling_rate = 22050
    num_mcep = 36
    frame_period = 5.0
    n_frames = 128

    # Training parameters
    num_iterations = 100000
    mini_batch_size = 8
    generator_learning_rate = 0.00020
    discriminator_learning_rate = 0.00010
    lambda_cycle = 10
    lambda_identity = 10
    lambda_triangle = 5
    lambda_backward = 5

    print('Loading cached data...')
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
    model = RelGAN(num_features=num_mcep, num_domains=num_domains, batch_size=mini_batch_size, log_dir=log_dir)

    ckpt = tf.train.get_checkpoint_state(os.path.join('experiments', model_name, 'checkpoints'))

    if ckpt:
        #last_model = ckpt.all_model_checkpoint_paths[1]
        last_model = ckpt.model_checkpoint_path
        print("loading {}".format(last_model))
        model.load(filepath=last_model)
    else:
        print("checkpoints are not found")

    iteration = 1
    while iteration <= num_iterations:
        if(iteration%10000==0):
            lambda_triangle *= 0.9
            lambda_backward *= 0.9
        generator_learning_rate *=0.99999
        discriminator_learning_rate *=0.99999
        x, x2, x_atr, y, y_atr, z, z_atr = sample_train_data(dataset_A=coded_sps_norms, nBatch=mini_batch_size, num_mcep=num_mcep, n_frames=n_frames)

        x_labels = np.zeros([mini_batch_size, num_domains])
        y_labels = np.zeros([mini_batch_size, num_domains])
        z_labels = np.zeros([mini_batch_size, num_domains])
        for b in range(mini_batch_size):
            x_labels[b] = np.identity(num_domains)[x_atr[b]]
            y_labels[b] = np.identity(num_domains)[y_atr[b]]
            z_labels[b] = np.identity(num_domains)[z_atr[b]]

        rnd = np.random.randint(2)
        alp = np.random.uniform(0, 0.5, size=mini_batch_size) if rnd==0 else np.random.uniform(0.5, 1.0, size=mini_batch_size)

        generator_loss, discriminator_loss, gen_adv_loss, gen_cond_loss, gen_int_loss, gen_rec_loss, gen_self_loss, dis_adv_loss, dis_cond_loss, dis_int_loss, lossb, lossm, losst = model.train(input_A=x,
                input_A2=x2, input_B=y, input_C=z, label_A=x_labels, label_B=y_labels, label_C=z_labels,
                 alpha=alp, rand=rnd, lambda_cycle=lambda_cycle, lambda_identity=lambda_identity, lambda_triangle=lambda_triangle, lambda_backward=lambda_backward,
                 generator_learning_rate=generator_learning_rate,
                 discriminator_learning_rate=discriminator_learning_rate)

        if iteration % 10 == 0:
            print('Iteration: {:07d}, Generator Loss : {:.3f}, Discriminator Loss : {:.3f}'.format(iteration,
                                                                                                   generator_loss,
                                                                                                   discriminator_loss))
            print("d_a=%.3f, d_c=%.3f, d_i=%.3f"%(dis_adv_loss, dis_cond_loss, dis_int_loss))
            print("g_a=%.3f, g_c=%.3f, g_i=%.3f, g_r=%.3f, g_s=%.3f, g_b=%.3f, g_m=%.3f, g_t=%.3f"%(gen_adv_loss, gen_cond_loss, gen_int_loss, gen_rec_loss, gen_self_loss, lossb, lossm, losst))
        if iteration % 5000 == 0:
            print('Checkpointing...')
            model.save(directory=os.path.join('experiments', model_name, 'checkpoints'),
                       filename='{}_{}.ckpt'.format(model_name, iteration))

        if iteration % 1000 == 0 :
            for q in range(3):
                eval_dirs = os.listdir('datasets_val')
                assert len(eval_dirs) == num_domains
                x, x2, x_atr, y, y_atr, z, z_atr = sample_train_data(dataset_A=coded_sps_norms, nBatch=1, num_mcep=num_mcep, n_frames=n_frames)
                x_labels = np.zeros([1, num_domains])
                y_labels = np.zeros([1, num_domains])
                for b in range(1):
                    x_labels[b] = np.identity(num_domains)[x_atr[b]]
                    y_labels[b] = np.identity(num_domains)[y_atr[b]]
                x_atr = x_atr[0]
                y_atr = y_atr[0]
                eval_A_dir = os.path.join('datasets_val', eval_dirs[x_atr])
                print(eval_A_dir)
                for file in glob.glob(eval_A_dir + '/*.wav'):
                    alpha = np.random.uniform(0, 1, size=1) if q!=0 else np.ones(1)
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
                    coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm]), label_A=x_labels, label_B=y_labels, alpha=alpha)[0]
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
                    validation_A_output_dir = 'test'
                    os.makedirs(validation_A_output_dir, exist_ok=True)
                    librosa.output.write_wav(os.path.join(validation_A_output_dir, "{:06d}_{}_to_{}_{:.3f}_{}".format(iteration,x_atr,y_atr,alpha[0],os.path.basename(file))), wav_transformed,
                                             sampling_rate)

        iteration += 1

if __name__ == '__main__':
    main()
