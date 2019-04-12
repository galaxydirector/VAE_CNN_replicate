# reference to Github.com/shaohua0116/VAE_Tensorflow
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

class VAE:
	def __init__(self,learning_rate=1e-4,batch_size=32,no_z=16,mode='fully_connected'):
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.no_z = no_z

		tf.reset_default_graph()
		if mode == 'fully_connected':
			self.build_fully_connected()
		elif mode == 'CNN':
			self.build_CNN()

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def build_fully_connected(self):
		self.x = tf.placeholder(name='input_x',dtype=tf.float32,shape=[None,input_dim])

		# Encode
		# x -> z_mean, z_sigma 
		f1 = fully_connected(self.x,256,scope='enc_fc1',activation_fn=tf.nn.elu)
		f2 = fully_connected(f1,128,scope='enc_fc2',activation_fn=tf.nn.elu)
		f3 = fully_connected(f2,64,scope='enc_fc3',activation_fn=tf.nn.elu)

		self.z_mu = fully_connected(f3,self.no_z,scope='enc_fc4_mu',activation_fn=None)
		self.z_log_sigma_sq = fully_connected(f3,self.no_z,scope='enc_fc4_sigma',activation_fn=None)


		# z_mean, z_sigma, random_normal -> z
		eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq),mean=0,stddev=1,dtype=tf.float32)
		self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

		# Decode
		# z -> x_hat
		g1 = fully_connected(self.z,64,scope='gen_fc1',activation_fn=tf.nn.elu)
		g2 = fully_connected(g1,128,scope='gen_fc2',activation_fn=tf.nn.elu)
		g3 = fully_connected(g2,256,scope='gen_fc3',activation_fn=tf.nn.elu)

		self.x_hat = fully_connected(g3,input_dim,scope='gen_out',activation_fn=tf.sigmoid)

		# Loss
		# Loss1: Reconstruction Loss
		# Cross Entropy of original and x_hat
		# proofs
		# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
		epsilon = 1e-10
		recon_loss = -tf.reduce_sum(self.x*tf.log(epsilon+self.x_hat)+(1-self.x)*tf.log(epsilon+1-self.x_hat),
			axis=1)
		self.recon_loss = tf.reduce_mean(recon_loss)



		# Loss2: Latent loss KL(q(z|x)||p(z))
		# KL divergence 
		# p(z) is a N(0,1)
		latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq),
            axis=1)
		self.latent_loss = tf.reduce_mean(latent_loss)

		self.total_loss = self.recon_loss + self.latent_loss
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)

		self.losses = {
			'recon_loss': self.recon_loss,
			'latent_loss': self.latent_loss,
			'total_loss': self.total_loss
		}

	# one training pass
	def run_single_step(self,x):
		_,losses = self.sess.run(
			[self.train_op, self.losses],
			feed_dict = {self.x:x})
		return losses

	# x -> x_hat
	def reconstructor(self,x):
		x_hat = self.sess.run(self.x_hat,
			feed_dict = {self.x:x})
		return x_hat

	# z -> x
	def generator(self,z):
		x_hat = self.sess.run(self.x_hat,
			feed_dict = {self.z:z})
		return x_hat

	def transformer(self,x):
		z = self.sess.run(self.z,feed_dict={self.x:x})
		return z





