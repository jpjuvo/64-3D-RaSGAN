import tensorflow as tf 
import os
import sys 
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import random 
import glob
import argparse
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser(description='3D-GAN implementation for 64*64*64 voxel output')
parser.add_argument('-n','--name', default='Test', help='The name of the current experiment, this will be used to create folders and save models.')
parser.add_argument('-d','--data', default='ModelNet10/chair/train', help ='The location of the voxel grid training models. (default=ModelNet40/chair/train)' )
parser.add_argument('-e','--epochs', default=5000, help ='The number of epochs to run for (default=5000)', type=int)
parser.add_argument('-b','--batchsize', default=24, help ='The batch size. (default=24)', type=int)
parser.add_argument('-sample', default= 10, help='How often generated obejcts are sampled and saved.', type= int)
parser.add_argument('-save', default= 10, help='How often the network models are saved.', type= int)
parser.add_argument('-l', '--load', default= False, help='Indicates if a previously loaded model should be loaded.', action = 'store_true')
parser.add_argument('-le', '--load_epoch', default= '', help='The epoch to number to be loaded from.', type=str)
parser.add_argument('-graph', default= 10, help='How often the discriminator loss and the reconstruction loss graphs are saved.', type= int)
parser.add_argument('-glr','--generator_learning_rate', default=0.0001, help ='The generator learning rate.', type=float)
parser.add_argument('-dlr','--discriminator_learning_rate', default=0.000025, help ='The discriminator learning rate.', type=float)
args = parser.parse_args()

checkpoint_dir = "checkpoint/" + args.name +'/'
save_dir =  "savepoint/" + args.name +'/'
output_size = 64
batchSize = args.batchsize

###########################################
################### MODELS ################
###########################################

def Deconv(inputs, f_dim_in, dim, net, batch_size, f_dim_out = None, stride = 2 ):
	if f_dim_out is None: 
		f_dim_out = int(f_dim_in/2) 
	return tl.layers.DeConv3dLayer(inputs,
								shape = [4, 4, 4, f_dim_out, f_dim_in],
								output_shape = [batch_size, dim, dim, dim, f_dim_out],
								strides=[1, stride, stride, stride, 1],
								W_init = tf.random_normal_initializer(stddev=0.02),
								act=tf.identity, name='g/net_' + net + '/deconv')
								
def Conv3D(inputs, f_dim_out, net, f_dim_in = None, batch_norm = False, is_train = True):
	if f_dim_in is None: 
		f_dim_in = int(f_dim_out/2)
	layer = tl.layers.Conv3dLayer(inputs, 
									shape=[4, 4, 4, f_dim_in, f_dim_out],
									W_init = tf.random_normal_initializer(stddev=0.02),
									strides=[1, 2, 2, 2, 1], name= 'd/net_' + net + '/conv')
	if batch_norm: 
		return tl.layers.BatchNormLayer(layer, is_train=is_train, name='d/net_' + net + '/batch_norm')
	else:
		return layer

def generator_64(inputs, is_train=True, reuse=False, batch_size = 128, sig = False):
	output_size, half, forth, eighth, sixteenth = 64, 32, 16, 8, 4
	gf_dim = 512 # Dimension of gen filters in first conv layer
	with tf.variable_scope("gen", reuse=reuse) as vs:

		net_0 = tl.layers.InputLayer(inputs, name='g/net_0/in')

		net_1 = tl.layers.DenseLayer(net_0, n_units = gf_dim*sixteenth*sixteenth*sixteenth, W_init = tf.random_normal_initializer(stddev=0.02), act = tf.identity, name='g/net_1/dense')
		net_1 = tl.layers.ReshapeLayer(net_1, shape = [-1, sixteenth, sixteenth, sixteenth, gf_dim], name='g/net_1/reshape')
		net_1 = tl.layers.BatchNormLayer(net_1, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='g/net_1/batch_norm')
		net_1.outputs = tf.nn.relu(net_1.outputs, name='g/net_1/relu')

		net_2 = Deconv(net_1, gf_dim, eighth, '2', batch_size) 
		net_2 = tl.layers.BatchNormLayer(net_2, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='g/net_2/batch_norm')
		net_2.outputs = tf.nn.relu(net_2.outputs, name='g/net_2/relu')

		net_3 = Deconv(net_2, int(gf_dim/2), forth, '3', batch_size)
		net_3 = tl.layers.BatchNormLayer(net_3, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='g/net_3/batch_norm')
		net_3.outputs = tf.nn.relu(net_3.outputs, name='g/net_3/relu')
		
		net_4 = Deconv(net_3, int(gf_dim/4), half, '4', batch_size)
		net_4 = tl.layers.BatchNormLayer(net_4, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='g/net_4/batch_norm')
		net_4.outputs = tf.nn.relu(net_4.outputs, name='g/net_4/relu')
	   
		net_5 = Deconv(net_4, int(gf_dim/8), output_size, '5', batch_size, f_dim_out = 1)
		net_5.outputs = tf.reshape(net_5.outputs,[batch_size,output_size,output_size,output_size])
		if sig: 
			net_5.outputs = tf.nn.sigmoid(net_5.outputs)
		else: 
			net_5.outputs = tf.nn.tanh(net_5.outputs)
		
		return net_5, net_5.outputs

def discriminator(inputs ,output_size, sig = False, is_train=True, reuse=False, batch_size=128, output_units= 1):
	inputs = tf.reshape(inputs,[batch_size,output_size,output_size,output_size,1])
	df_dim = output_size # Dimension of discrim filters in first conv layer

	with tf.variable_scope("dis", reuse=reuse) as vs:
	
		net_0 = tl.layers.InputLayer(inputs, name='d/net_0/in')

		net_1 = Conv3D(net_0, df_dim, '1', f_dim_in = 1 , batch_norm = False ) 
		net_1.outputs = tf.nn.leaky_relu(net_1.outputs, alpha=0.2, name='d/net_1/lrelu')
		
		net_2 = Conv3D(net_1, int(df_dim*2), '2', batch_norm = True, is_train = is_train,) 
		net_2.outputs = tf.nn.leaky_relu(net_2.outputs, alpha=0.2, name='d/net_2/lrelu')
		
		net_3 = Conv3D(net_2, int(df_dim*4), '3', batch_norm = True, is_train = is_train)  
		net_3.outputs = tf.nn.leaky_relu(net_3.outputs, alpha=0.2, name='d/net_3/lrelu')
		
		net_4 = Conv3D(net_3, int(df_dim*8), '4', batch_norm = True, is_train = is_train)   
		net_4.outputs = tf.nn.leaky_relu(net_4.outputs, alpha=0.2, name='d/net_4/lrelu')
		
		net_5 = FlattenLayer(net_4, name='d/net_5/flatten')
		net_5 = tl.layers.DenseLayer(net_5, n_units=output_units, act=tf.identity,
										W_init = tf.random_normal_initializer(stddev=0.02),
										name='d/net_5/dense')
		if sig: 
			return net_5, tf.nn.sigmoid(net_5.outputs)
		else: 
			return net_5, net_5.outputs 

###########################################
#################### Utils ################
###########################################

def make_inputs_raw(file_batch):
	dt = np.dtype((np.uint8, (64,64,64)))
	models = [np.fromfile(f,dtype=dt).reshape((64,64,64)) for f in file_batch]
	models = np.array(models)
	start_time = time.time()
	return models, start_time
			
def load_networks(checkpoint_dir, sess, net_g, net_d, epoch = ''): 
	print("[*] Loading checkpoints...")
	if len(epoch) >=1: epoch = '_' + epoch
	# load the latest checkpoints
	net_g_name = os.path.join(checkpoint_dir, 'net_g'+epoch+'.npz')
	net_d_name = os.path.join(checkpoint_dir, 'net_d'+epoch+'.npz')
	
	if not (os.path.exists(net_g_name) and os.path.exists(net_d_name)):
		print("[!] Loading checkpoints failed!")
	else:
		net_g_loaded_params = tl.files.load_npz(name=net_g_name)
		net_d_loaded_params = tl.files.load_npz(name=net_d_name)
		tl.files.assign_params(sess, net_g_loaded_params, net_g)
		tl.files.assign_params(sess, net_d_loaded_params, net_d)
		print("[*] Loading Generator and Discriminator checkpoints SUCCESS!")
			
def save_networks(checkpoint_dir, sess, net_g, net_d, epoch):
	print("[*] Saving checkpoints...")
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	# this saves as the latest version location
	net_g_name = os.path.join(checkpoint_dir, 'net_g.npz')
	net_d_name = os.path.join(checkpoint_dir, 'net_d.npz')
	# this saves as a backlog of models
	net_g_iter_name = os.path.join(checkpoint_dir, 'net_g_%d.npz' % epoch)
	net_d_iter_name = os.path.join(checkpoint_dir, 'net_d_%d.npz' % epoch)
	tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
	tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
	tl.files.save_npz(net_g.all_params, name=net_g_iter_name, sess=sess)
	tl.files.save_npz(net_d.all_params, name=net_d_iter_name, sess=sess)

	print("[*] Saving checkpoints SUCCESS!")
	
def save_voxels(save_dir, models, epock): 
	print('Saving the model')
	#save only one from batch per epoch to save space
	np.save(save_dir+str(epock)  , models[0])
	
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
	from math import factorial
	try:
		window_size = np.abs(np.int(window_size))
		order = np.abs(np.int(order))
	except ValueError:
		raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
	order_range = range(order+1)
	half_window = (window_size -1) // 2
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m[::-1], y, mode='valid')
	
def render_graphs(save_dir,epoch, track_g_loss, track_d_loss, epoch_arr): 
	if not os.path.exists(save_dir+'/plots/'):
		os.makedirs(save_dir+'/plots/')
	if len(track_d_loss)> 51: 
		#smoothed_d_loss = savitzky_golay(track_d_loss, 51, 3)
		#smoothed_g_loss = savitzky_golay(track_g_loss, 51, 3)
		plt.plot(epoch_arr, track_d_loss, color='blue', alpha=0.5)
		#plt.plot(epoch_arr, smoothed_d_loss, color = 'navy', alpha=0.5)
		plt.plot(epoch_arr, track_g_loss, color='red', alpha=0.5)
		#plt.plot(epoch_arr, smoothed_g_loss, color = 'crimson', alpha=0.5)
		plt.legend(('Discriminator\'s loss','Generator\'s loss'), loc='upper right')
		plt.title('64-3D-GAN')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.grid(True)
		plt.savefig(save_dir+'/plots/' + str(epoch)+'.png' )
		plt.clf()

def save_values(save_dir,track_g_loss, track_d_loss, epoch_arr):
	np.save(save_dir+'/plots/track_g_loss', track_g_loss)
	np.save(save_dir+'/plots/track_d_loss', track_d_loss)
	np.save(save_dir+'/plots/epochs', epoch_arr)
	
def load_values(save_dir):
	outputs = []
	outputs.append(list(np.load(save_dir+'/plots/track_g_loss.npy')))
	outputs.append(list(np.load(save_dir+'/plots/track_d_loss.npy')))
	outputs.append(list(np.load(save_dir+'/plots/epochs.npy')))
	return outputs
	
def cal_acc(zeros,ones): 
	accuracy = 0.0
	for example in zeros:
		if not np.isnan(example[0]):
			if example[0]<0.5:
				accuracy += 1.0
	
	for example in ones:
		if not np.isnan(example[0]):
			if example[0]>0.5:
				accuracy += 1.0 
	accuracy = accuracy/(float(len(zeros) + len(ones))) 
	print('The accuracy of the discriminator is: ' + str(accuracy))
	return accuracy

###########################################
######### make directories ################
###########################################

if not os.path.exists(checkpoint_dir):
	os.makedirs(checkpoint_dir)

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

#######################################
########### inputs  ###################
#######################################

z = tf.random_normal((batchSize, 200), 0, 1)
real_models =  tf.placeholder(tf.float32, [batchSize, output_size, output_size, output_size] , name='real_models')
a = tf.Print(z, [z], message="This is a: ")

#######################################################
########## network computations #######################
#######################################################

#used for training generator
net_g, G_train =  generator_64(z, is_train=True, reuse = False, sig= True, batch_size=batchSize)

net_d , D_fake      = discriminator(G_train, output_size, batch_size= batchSize, sig = True, is_train = True, reuse = False)
net_d2, D_legit     = discriminator(real_models,  output_size, batch_size= batchSize, sig = True, is_train= True, reuse = True)
net_d2, D_eval      = discriminator(real_models,  output_size, batch_size= batchSize, sig = True, is_train= False, reuse = True) # this is for deciding weather to train the discriminator

##########################################
########### Loss calculations ############
##########################################

d_loss = -tf.reduce_mean(tf.log(D_legit) + tf.log(1. - D_fake))
g_loss = -tf.reduce_mean(tf.log(D_fake))

##########################################
############## Optimization ##############
##########################################

g_vars = net_g.all_params   
d_vars = net_d.all_params  

g_vars = tl.layers.get_variables_with_name('gen', True, True)   
d_vars = tl.layers.get_variables_with_name('dis', True, True)

d_optim = tf.train.AdamOptimizer(args.discriminator_learning_rate, beta1=0.5).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(args.generator_learning_rate, beta1=0.5).minimize(g_loss, var_list=g_vars)

#######################################
############# Training ################
#######################################

#Allocate only what is needed of gpu memory and then grow
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess=tf.Session(config=config)
sess=tf.Session()
sess.run(tf.global_variables_initializer())

# load checkpoints
if args.load: 
	load_networks(checkpoint_dir, sess, net_g, net_d, epoch = args.load_epoch)
	track_g_loss, track_d_loss, epoch_arr = load_values(save_dir)
else:     
	track_g_loss, track_d_loss, epoch_arr = [],[],[]

#Get training model files
files = glob.glob(args.data + '/*.raw')
if(len(files) == 0):
	print('Could not find any raw voxel grid files in ' + args.data)
	print('Please check that the .off files are converted with convert_data.py')
	exit(1)

if (len(files) < batchSize):
	print('Batch size is larger than sample pool size. No worries, adjusting bacth size to ' + str(len(files)))
	batchSize = len(files)
	
Train_Dis = True 

if len(args.load_epoch)>1: 
	start = int(args.load_epoch)
else: 
	start = 0

for epoch in range(start, args.epochs):
	random.shuffle(files)
	for idx in range(0, int(len(files)/batchSize)):
		file_batch = files[idx*batchSize:(idx+1)*batchSize]
		models, start_time = make_inputs_raw(file_batch)
		#training the discriminator 
		if Train_Dis: 
			errD,_,ones = sess.run([d_loss, d_optim, D_legit] ,feed_dict={real_models: models}) 
		else: 
			ones = sess.run([D_eval] ,feed_dict={real_models: models}) 
		errG,_,zeros,objects = sess.run([g_loss, g_optim, D_fake, G_train], feed_dict={})    
		Train_Dis = (cal_acc(zeros,ones)<0.80)# only train discriminator at certain level of accuracy 
		
		#Append arrays
		track_g_loss.append(errG)
		track_d_loss.append(errD)
		epoch_arr.append(epoch)
		
		print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
					% (epoch, args.epochs, idx, int(len(files)/batchSize) , time.time() - start_time, errD, errG))
	#saving the model 
	if np.mod(epoch, args.save) == 0:
		save_networks(checkpoint_dir,sess, net_g, net_d, epoch)
	#saving generated objects
	if np.mod(epoch, args.sample ) == 0:     
		save_voxels(save_dir,objects, epoch )
	#saving learning info 
	if np.mod(epoch, args.graph) == 0: 
		render_graphs(save_dir,epoch, track_g_loss, track_d_loss, epoch_arr) #this will only work after a 50 iterations to allow for proper averaging 
		save_values(save_dir, track_g_loss, track_d_loss, epoch_arr)



    
