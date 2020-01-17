import os
import tensorflow as tf
from models.modules import PatchGanDiscriminator, Generator212
from models.utils import l1_loss, l2_loss
from datetime import datetime


class RelGAN(object):

    def __init__(self, num_features, num_domains, batch_size=1, discriminator=PatchGanDiscriminator, generator=Generator212,
                 mode='train', log_dir='./log'):

        self.num_features = num_features
        self.num_domains = num_domains
        self.batch_size = batch_size
        self.input_shape = [None, num_features, None]  # [batch_size, num_features, num_frames]
        self.label_shape = [None, num_domains]

        self.discriminator = discriminator
        self.generator = generator
        self.mode = mode

        self.build_model()
        self.optimizer_initializer()

        self.saver = tf.train.Saver(max_to_keep=10)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            self.writer = tf.summary.FileWriter(self.log_dir)
            self.generator_summaries, self.discriminator_summaries = self.summary()

    def build_model(self):

        # Placeholders for real training samples
        self.input_A_real = tf.placeholder(tf.float32, shape=self.input_shape, name='input_A_real')
        self.input_A2_real = tf.placeholder(tf.float32, shape=self.input_shape, name='input_A_real')
        self.input_B_real = tf.placeholder(tf.float32, shape=self.input_shape, name='input_B_real')
        self.input_C_real = tf.placeholder(tf.float32, shape=self.input_shape, name='input_B_real')
        # Placeholders for label of real training samples
        self.input_A_label = tf.placeholder(tf.float32, shape=self.label_shape, name='input_A_label')
        self.input_B_label = tf.placeholder(tf.float32, shape=self.label_shape, name='input_B_label')
        self.input_C_label = tf.placeholder(tf.float32, shape=self.label_shape, name='input_C_label')
        # placeholders for alpha
        self.rnd = tf.placeholder(tf.float32, [])
        self.alpha = tf.placeholder(tf.float32, shape=[None])
        self.alpha_1 = tf.reshape(self.alpha, [-1, 1])
        # Placeholders for fake generated samples
        self.input_A_fake = tf.placeholder(tf.float32, shape=self.input_shape, name='input_A_fake')
        self.input_B_fake = tf.placeholder(tf.float32, shape=self.input_shape, name='input_B_fake')
        # Placeholders for cycle generated samples
        self.input_A_cycle = tf.placeholder(tf.float32, shape=self.input_shape, name='input_A_cycle')
        self.input_B_cycle = tf.placeholder(tf.float32, shape=self.input_shape, name='input_B_cycle')
        # Placeholder for test samples
        self.input_A_test = tf.placeholder(tf.float32, shape=self.input_shape, name='input_A_test')

        self.vector_A2B = self.input_B_label - self.input_A_label
        self.vector_C2B = self.input_B_label - self.input_C_label
        self.vector_A2C = self.input_C_label - self.input_A_label

        self.lambda_conditional = 1
        self.lambda_interp = 10
        #self.lambda_forward = 10
        self.lambda_backward = tf.placeholder(tf.float32, None, name='lambda_backward')
        self.lambda_triangle = tf.placeholder(tf.float32, None, name='lambda_triangle')
        self.lambda_mode_seeking = 1

        self.generation_B = self.generator(inputs=self.input_A_real, vec=self.vector_A2B, num_domains=self.num_domains,
                                            dim=self.num_features, batch_size=self.batch_size,
                                           reuse=False, scope_name='generator_A2B')

        self.generation_B2 = self.generator(inputs=self.input_A2_real, vec=self.vector_A2B, num_domains=self.num_domains,
                                            dim=self.num_features, batch_size=self.batch_size,
                                           reuse=True, scope_name='generator_A2B')

        self.cycle_A = self.generator(inputs=self.generation_B, vec=-self.vector_A2B, num_domains=self.num_domains,
                                        dim=self.num_features, batch_size=self.batch_size,
                                      reuse=True, scope_name='generator_A2B')

        self.generation_A_identity = self.generator(inputs=self.input_A_real, vec=self.vector_A2B-self.vector_A2B,
                                        num_domains=self.num_domains, dim=self.num_features,
                                                    batch_size=self.batch_size, reuse=True,
                                                    scope_name='generator_A2B')

        self.generation_alp = self.generator(inputs=self.input_A_real, vec=self.vector_A2B*self.alpha_1,
                                        num_domains=self.num_domains, dim=self.num_features, batch_size=self.batch_size,
                                           reuse=True,
                                           scope_name='generator_A2B')

        self.generation_alp_forward = self.generator(inputs=self.generation_alp, vec=self.vector_A2B*(1-self.alpha_1),
                                        num_domains=self.num_domains, dim=self.num_features, batch_size=self.batch_size,
                                           reuse=True,
                                           scope_name='generator_A2B')

        self.generation_alp_backward = self.generator(inputs=self.generation_alp, vec=-self.vector_A2B*self.alpha_1,
                                        num_domains=self.num_domains, dim=self.num_features, batch_size=self.batch_size,
                                           reuse=True,
                                           scope_name='generator_A2B')

        self.generation_C = self.generator(inputs=self.generation_B, vec=-self.vector_C2B, num_domains=self.num_domains,
                                            dim=self.num_features, batch_size=self.batch_size,
                                           reuse=True, scope_name='generator_A2B')

        self.generation_A = self.generator(inputs=self.generation_C, vec=-self.vector_A2C, num_domains=self.num_domains,
                                            dim=self.num_features, batch_size=self.batch_size,
                                           reuse=True, scope_name='generator_A2B')

        # One-step discriminator

        self.discrimination_B_fake = self.discriminator(inputs_A=self.generation_B, inputs_B=None,vec=None,
                                                        num_domains=self.num_domains, reuse=[False,False],
                                                        scope_name='discriminator_B', method='adversarial')

        self.discrimination_alp_fake = self.discriminator(inputs_A=self.generation_alp, inputs_B=None,vec=None,
                                                        num_domains=self.num_domains, reuse=[True,True],
                                                        scope_name='discriminator_B', method='adversarial')

        # Two-step discriminator

        self.discrimination_A_dot_fake = self.discriminator(inputs_A=self.cycle_A, inputs_B=None,vec=None,
                                                        num_domains=self.num_domains, reuse=[True,True],
                                                            scope_name='discriminator_B', method='adversarial')


        # Cycle loss
        self.cycle_loss = l1_loss(y=self.input_A_real, y_hat=self.cycle_A)

        # Identity loss
        self.identity_loss = l1_loss(y=self.input_A_real, y_hat=self.generation_A_identity)

        # Forward loss
        self.forward_loss = l1_loss(y=self.generation_B, y_hat=self.generation_alp_forward)

        # Backward loss
        self.backward_loss = l1_loss(y=self.input_A_real, y_hat=self.generation_alp_backward)

        # Mode seeking Loss
        self.mode_seeking_loss = tf.divide(l1_loss(y=self.input_A_real, y_hat=self.input_A2_real), l1_loss(y=self.generation_B, y_hat=self.generation_B2)+1e-12)

        # Triangle Loss
        self.triangle_loss = l1_loss(y=self.input_A_real, y_hat=self.generation_A)

        # Place holder for lambda_cycle and lambda_identity
        self.lambda_cycle = tf.placeholder(tf.float32, None, name='lambda_cycle')
        self.lambda_identity = tf.placeholder(tf.float32, None, name='lambda_identity')

        # ------------------------------- Generator and Discriminator loss
        # Generator wants to fool discriminator
        self.generator_loss_A2B = l2_loss(y=tf.ones_like(self.discrimination_B_fake), y_hat=self.discrimination_B_fake) #+ l2_loss(y=tf.ones_like(self.discrimination_alp_fake), y_hat=self.discrimination_alp_fake)
        #self.generator_loss_alp =


        # Two-step generator loss

        self.two_step_generator_loss_A = l2_loss(y=tf.ones_like(self.discrimination_A_dot_fake),
                                                 y_hat=self.discrimination_A_dot_fake)


        # One-step
        self.discrimination_input_B_real = self.discriminator(inputs_A=self.input_B_real, inputs_B=None,vec=None,
                                                        num_domains=self.num_domains, reuse=[True,True],
                                                        scope_name='discriminator_B', method='adversarial')

        self.discrimination_input_B_fake = self.discriminator(inputs_A=self.input_B_fake, inputs_B=None,vec=None,
                                                        num_domains=self.num_domains, reuse=[True,True],
                                                        scope_name='discriminator_B', method='adversarial')

        # Two-step
        self.discrimination_input_A_dot_real = self.discriminator(inputs_A=self.input_A_real,inputs_B=None,vec=None,
                                                        num_domains=self.num_domains, reuse=[True,True],
                                                            scope_name='discriminator_B', method='adversarial')

        self.discrimination_input_A_dot_fake = self.discriminator(inputs_A=self.input_A_cycle, inputs_B=None,vec=None,
                                                        num_domains=self.num_domains, reuse=[True,True],
                                                            scope_name='discriminator_B', method='adversarial')

        # Discriminator wants to classify real and fake correctly
        self.discriminator_loss_input_B_real = l2_loss(y=tf.ones_like(self.discrimination_input_B_real),
                                                       y_hat=self.discrimination_input_B_real)
        self.discriminator_loss_input_B_fake = l2_loss(y=tf.zeros_like(self.discrimination_input_B_fake),
                                                       y_hat=self.discrimination_input_B_fake)
        self.discriminator_loss_alp = l2_loss(y=tf.zeros_like(self.discrimination_alp_fake), y_hat=self.discrimination_alp_fake)
        self.discriminator_loss_B = (self.discriminator_loss_input_B_real + self.discriminator_loss_input_B_fake) / 2
        #self.discriminator_loss_B = (self.discriminator_loss_input_B_real + self.discriminator_loss_input_B_fake + self.discriminator_loss_alp) / 3

        # Two-step discriminator loss
        self.two_step_discriminator_loss_input_A_real = l2_loss(y=tf.ones_like(self.discrimination_input_A_dot_real),
                                                                y_hat=self.discrimination_input_A_dot_real)
        self.two_step_discriminator_loss_input_A_fake = l2_loss(y=tf.zeros_like(self.discrimination_input_A_dot_fake),
                                                                y_hat=self.discrimination_input_A_dot_fake)
        self.two_step_discriminator_loss_A = (self.two_step_discriminator_loss_input_A_real +
                                              self.two_step_discriminator_loss_input_A_fake) / 2

        # Conditional adversarial Loss

        self.sr = self.discriminator(inputs_A=self.input_A_real,inputs_B=self.input_B_real,vec=self.vector_A2B,
                                                        num_domains=self.num_domains, reuse=[True,False],
                                                            scope_name='discriminator_B', method='matching')
        self.sf = self.discriminator(inputs_A=self.input_A_real,inputs_B=self.generation_B,vec=self.vector_A2B,
                                                        num_domains=self.num_domains, reuse=[True,True],
                                                            scope_name='discriminator_B', method='matching')
        self.w1 = self.discriminator(inputs_A=self.input_C_real,inputs_B=self.input_B_real,vec=self.vector_A2B,
                                                        num_domains=self.num_domains, reuse=[True,True],
                                                            scope_name='discriminator_B', method='matching')
        self.w2 = self.discriminator(inputs_A=self.input_A_real,inputs_B=self.input_B_real,vec=self.vector_C2B,
                                                        num_domains=self.num_domains, reuse=[True,True],
                                                            scope_name='discriminator_B', method='matching')
        self.w3 = self.discriminator(inputs_A=self.input_A_real,inputs_B=self.input_B_real,vec=self.vector_A2C,
                                                        num_domains=self.num_domains, reuse=[True,True],
                                                            scope_name='discriminator_B', method='matching')
        self.w4 = self.discriminator(inputs_A=self.input_A_real,inputs_B=self.input_C_real,vec=self.vector_A2B,
                                                        num_domains=self.num_domains, reuse=[True,True],
                                                            scope_name='discriminator_B', method='matching')

        self.discriminator_loss_conditional_sr = l2_loss(y=tf.ones_like(self.sr), y_hat=self.sr)
        self.discriminator_loss_conditional_sf = l2_loss(y=tf.zeros_like(self.sf), y_hat=self.sf)
        self.discriminator_loss_conditional_w1 = l2_loss(y=tf.zeros_like(self.w1), y_hat=self.w1)
        self.discriminator_loss_conditional_w2 = l2_loss(y=tf.zeros_like(self.w2), y_hat=self.w2)
        self.discriminator_loss_conditional_w3 = l2_loss(y=tf.zeros_like(self.w3), y_hat=self.w3)
        self.discriminator_loss_conditional_w4 = l2_loss(y=tf.zeros_like(self.w4), y_hat=self.w4)
        self.discriminator_loss_conditional = self.discriminator_loss_conditional_sr + self.discriminator_loss_conditional_sf + \
                                            self.discriminator_loss_conditional_w1 + self.discriminator_loss_conditional_w2 + \
                                            self.discriminator_loss_conditional_w3 + self.discriminator_loss_conditional_w4

        self.generator_loss_conditional_sf = l2_loss(y=tf.ones_like(self.sf), y_hat=self.sf)

        # Interpolation Loss
        self.interpolate_identity = self.discriminator(inputs_A=self.generation_A_identity,inputs_B=None,vec=None,
                                                        num_domains=self.num_domains, reuse=[True,False],
                                                            scope_name='discriminator_B', method='interpolation')
        self.interpolate_B = self.discriminator(inputs_A=self.generation_B,inputs_B=None,vec=None,
                                                        num_domains=self.num_domains, reuse=[True,True],
                                                            scope_name='discriminator_B', method='interpolation')
        self.interpolate_alp = self.discriminator(inputs_A=self.generation_alp,inputs_B=None,vec=None,
                                                        num_domains=self.num_domains, reuse=[True,True],
                                                            scope_name='discriminator_B', method='interpolation')

        self.discriminator_loss_interp_AB = l2_loss(y=tf.zeros_like(self.interpolate_identity), y_hat=self.interpolate_identity) if self.rnd==0 else l2_loss(y=tf.zeros_like(self.interpolate_B), y_hat=self.interpolate_B)
        #print(self.alpha_1<0.5)
        #print(self.interpolate_identity)
        #self.discriminator_loss_interp_AB = tf.where(self.alpha_1<0.5,self.interpolate_identity, self.interpolate_B)

        self.discriminator_loss_interp_alp = l2_loss(y=tf.ones_like(self.interpolate_alp)*tf.reshape(self.alpha_1,[-1,1,1,1]), y_hat=self.interpolate_alp) if self.rnd==0 else  l2_loss(y=tf.ones_like(self.interpolate_alp)*tf.reshape(1-self.alpha_1,[-1,1,1,1]), y_hat=self.interpolate_alp)
        #self.discriminator_loss_interp_alp = tf.where(self.alpha_1<0.5,l2_loss(y=tf.ones_like(self.interpolate_alp)*tf.reshape(self.alpha_1,[-1,1,1,1]), y_hat=self.interpolate_alp),l2_loss(y=tf.ones_like(self.interpolate_alp)*tf.reshape(1-self.alpha_1,[-1,1,1,1]), y_hat=self.interpolate_alp))

        self.discriminator_loss_interp = self.discriminator_loss_interp_AB + self.discriminator_loss_interp_alp

        self.generator_loss_interp_alp = l2_loss(y=tf.zeros_like(self.interpolate_alp), y_hat=self.interpolate_alp)

        # Merge the generator losses

        self.generator_loss = self.generator_loss_A2B + self.lambda_backward * self.backward_loss + self.mode_seeking_loss + \
                              self.lambda_cycle * self.cycle_loss + self.lambda_identity * self.identity_loss + self.lambda_triangle*self.triangle_loss + \
                              self.lambda_conditional * self.generator_loss_conditional_sf + self.lambda_interp * self.generator_loss_interp_alp



        #self.generator_loss = self.lambda_identity * self.identity_loss
        # Merge the discriminator Losses

        self.discriminator_loss = self.discriminator_loss_B +  \
                                    self.lambda_interp * self.discriminator_loss_interp + self.lambda_conditional * self.discriminator_loss_conditional

        #self.discriminator_loss = self.discriminator_loss_B
        # Categorize variables because we have to optimize the two sets of the variables separately
        trainable_variables = tf.trainable_variables()
        self.discriminator_vars = [var for var in trainable_variables if 'discriminator' in var.name]
        self.generator_vars = [var for var in trainable_variables if 'generator' in var.name]
        # for var in t_vars: print(var.name)

        # Reserved for test
        self.generation_B_test = self.generator(inputs=self.input_A_test, vec=self.vector_A2B*self.alpha_1, num_domains=self.num_domains,
                                            dim=self.num_features, batch_size=1,
                                           reuse=True, scope_name='generator_A2B')

    def optimizer_initializer(self):

        self.generator_learning_rate = tf.placeholder(tf.float32, None, name='generator_learning_rate')
        self.discriminator_learning_rate = tf.placeholder(tf.float32, None, name='discriminator_learning_rate')
        self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.discriminator_learning_rate,
                                                              beta1=0.5).minimize(self.discriminator_loss,
                                                                                  var_list=self.discriminator_vars)
        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.generator_learning_rate,
                                                          beta1=0.5).minimize(self.generator_loss,
                                                                              var_list=self.generator_vars)

    def train(self, input_A, input_A2, input_B, input_C, label_A, label_B, label_C, alpha, rand, lambda_cycle, lambda_identity, lambda_backward, lambda_triangle,
                generator_learning_rate, discriminator_learning_rate):

        generation_B, cycle_A, generator_loss, _, generator_summaries, gen_adv_loss, gen_cond_loss, gen_int_loss, gen_rec_loss, gen_self_loss,forlos,backlos,modelos,trilos = self.sess.run(
            [self.generation_B, self.cycle_A, self.generator_loss,
             self.generator_optimizer, self.generator_summaries, self.generator_loss_A2B, self.generator_loss_conditional_sf,
             self.generator_loss_interp_alp,self.cycle_loss, self.identity_loss, self.forward_loss, self.backward_loss, self.mode_seeking_loss,self.triangle_loss],
            feed_dict={self.lambda_cycle: lambda_cycle, self.lambda_identity: lambda_identity, self.lambda_backward:lambda_backward, self.lambda_triangle:lambda_triangle,
                        self.input_A_real: input_A, self.input_A2_real:input_A2, self.input_A_label: label_A, self.input_B_label: label_B, self.input_C_label:label_C,
                        self.rnd: rand, self.alpha: alpha, self.generator_learning_rate: generator_learning_rate})

        self.writer.add_summary(generator_summaries, self.train_step)

        discriminator_loss, _, discriminator_summaries , dis_adv_loss, dis_cond_loss, dis_int_loss, intab, intal = self.sess.run(
            [self.discriminator_loss, self.discriminator_optimizer, self.discriminator_summaries, self.discriminator_loss_B,
            self.discriminator_loss_conditional, self.discriminator_loss_interp, self.discriminator_loss_interp_AB, self.discriminator_loss_interp_alp],
            feed_dict={self.input_A_real: input_A, self.input_B_real: input_B, self.input_C_real: input_C,
                        self.input_A_label: label_A, self.input_B_label: label_B, self.input_C_label: label_C,
                        self.rnd: rand, self.alpha: alpha, self.discriminator_learning_rate: discriminator_learning_rate,
                        self.input_B_fake: generation_B, self.input_A_cycle: cycle_A})

        self.writer.add_summary(discriminator_summaries, self.train_step)

        self.train_step += 1

        return generator_loss, discriminator_loss, gen_adv_loss, gen_cond_loss, gen_int_loss, gen_rec_loss, gen_self_loss, dis_adv_loss, dis_cond_loss, dis_int_loss ,intab,intal, forlos, backlos, modelos,trilos

    def test(self, inputs, label_A, label_B, alpha):

        generation = self.sess.run(self.generation_B_test, feed_dict={self.input_A_test: inputs,
                        self.input_A_label: label_A, self.input_B_label: label_B, self.alpha: alpha})

        return generation

    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))

        return os.path.join(directory, filename)

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)

    def summary(self):
        with tf.name_scope('generator_summaries'):
            cycle_loss_summary = tf.summary.scalar('cycle_loss', self.cycle_loss)
            identity_loss_summary = tf.summary.scalar('identity_loss', self.identity_loss)
            triangle_loss_summary = tf.summary.scalar('triangle_loss', self.triangle_loss)
            backward_loss_summary = tf.summary.scalar('backward_loss', self.backward_loss)
            interpolation_loss_summary = tf.summary.scalar('interpolation_loss', self.generator_loss_interp_alp)
            conditional_loss_summary = tf.summary.scalar('Conditional_loss', self.generator_loss_conditional_sf)
            generator_loss_A2B_summary = tf.summary.scalar('adversarial_loss', self.generator_loss_A2B)
            mode_seeking_loss_summary = tf.summary.scalar('mode_seeking_loss', self.mode_seeking_loss)
            generator_loss_summary = tf.summary.scalar('generator_loss', self.generator_loss)
            generator_summaries = tf.summary.merge(
                [cycle_loss_summary, identity_loss_summary, triangle_loss_summary,generator_loss_A2B_summary,
                 backward_loss_summary, interpolation_loss_summary, conditional_loss_summary,
                 mode_seeking_loss_summary, generator_loss_summary])

        with tf.name_scope('discriminator_summaries'):
            discriminator_loss_B_summary = tf.summary.scalar('discriminator_loss_B', self.discriminator_loss_B)
            discriminator_loss_conditional_summary = tf.summary.scalar('discriminator_loss_conditional', self.discriminator_loss_conditional)
            discriminator_loss_interp_summary = tf.summary.scalar('discriminator_loss_interp', self.discriminator_loss_interp)
            discriminator_loss_summary = tf.summary.scalar('discriminator_loss', self.discriminator_loss)
            discriminator_summaries = tf.summary.merge(
                [discriminator_loss_B_summary, discriminator_loss_conditional_summary,
                discriminator_loss_interp_summary, discriminator_loss_summary])

        return generator_summaries, discriminator_summaries


if __name__ == '__main__':
    import numpy as np

    batch_size = 5
    #model = CycleGAN2(num_features=36, batch_size=batch_size)
    model = RelGAN(num_features=36, num_domains=4, batch_size=batch_size)
    gen = model.test(inputs=np.random.randn(batch_size, 36, 317), direction='A2B')
    print(gen.shape)
    print('Graph Compile Successeded.')
