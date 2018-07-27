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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time

parser = argparse.ArgumentParser(description='3D-GAN implementation for 64*64*64 voxel output')
parser.add_argument('-n','--name', default='Test', help='The name of the current experiment, this will be used to create folders and save models.')
parser.add_argument('-d','--data', default='ModelNet10/chair/train', help ='The location of the voxel grid training models. (default=ModelNet10/chair/train)' )
parser.add_argument('-v','--validation_data', default='', help ='Optional location of the voxel grid validation models. (for example ModelNet10/chair/test)' )
parser.add_argument('-e','--epochs', default=2500, help ='The number of epochs to run for (default=2500)', type=int)
parser.add_argument('-b','--batchsize', default=24, help ='The batch size. (default=24)', type=int)
parser.add_argument('-sample', default= 10, help='How often generated obejcts are sampled and saved.', type= int)
parser.add_argument('-save', default= 10, help='How often the network models are saved.', type= int)
parser.add_argument('-l', '--load', default= False, help='Indicates if a previously loaded model should be loaded.', action = 'store_true')
parser.add_argument('-le', '--load_epoch', default= '', help='The epoch to number to be loaded from.', type=str)
parser.add_argument('-graph', default= 10, help='How often the loss graphs are saved.', type= int)
parser.add_argument('-glr','--generator_learning_rate', default=0.0020, help ='The generator\'s learning rate.', type=float)
parser.add_argument('-dlr','--discriminator_learning_rate', default=0.00005, help ='The discriminator\'s learning rate.', type=float)
parser.add_argument('-graph3d', default= 10, help='How often the 3D graphs are saved.', type=int)
args = parser.parse_args()

checkpoint_dir = "checkpoint/" + args.name +'/'
save_dir =  "savepoint/" + args.name +'/'
graph_3d_dir = "savepoint/" + args.name +'/3D_graphs/'
output_size = 64
batchSize = args.batchsize
epsilon = 1e-14


###########################################
################### MODELS ################
###########################################

def Deconv(inputs, f_dim_in, dim, net, batch_size, f_dim_out = None, stride = 2):
	if f_dim_out is None: 
		f_dim_out = int(f_dim_in/2) 
	return tl.layers.DeConv3dLayer(inputs,
								shape = [4, 4, 4, f_dim_out, f_dim_in],
								output_shape = [batch_size, dim, dim, dim, f_dim_out],
								strides=[1, stride, stride, stride, 1],
								W_init = tf.random_normal_initializer(stddev=0.02),
								act=tf.identity, name='g/net_' + net + '/deconv')
								
def Conv3D(inputs, f_dim_out, net, f_dim_in = None, batch_norm = False, is_train = True, stride = 2):
	if f_dim_in is None: 
		f_dim_in = int(f_dim_out/2)
	layer = tl.layers.Conv3dLayer(inputs, 
									shape=[4, 4, 4, f_dim_in, f_dim_out],
									W_init = tf.random_normal_initializer(stddev=0.02),
									strides=[1, stride, stride, stride, 1], name= 'd/net_' + net + '/conv')
	if batch_norm: 
		return tl.layers.BatchNormLayer(layer, is_train=is_train, name='d/net_' + net + '/batch_norm')
	else:
		return layer

def generator_64(inputs, is_train=True, reuse=False, batch_size = 24, sig = False):
	output_size, half, forth, eighth, sixteenth = 64, 32, 16, 8, 4
	gf_dim = 512 # Dimension of gen filters in first conv layer
	with tf.variable_scope("gen", reuse=reuse) as vs:

		net_0 = tl.layers.InputLayer(inputs, name='g/net_0/in')
		#Fully connected
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
		#net_5.outputs = tf.nn.relu(net_5.outputs, name='g/net_5/relu')
		if sig: 
			net_5.outputs = tf.nn.sigmoid(net_5.outputs)
		#else: 
			#net_5.outputs = tf.nn.tanh(net_5.outputs)
		
		return net_5, net_5.outputs

def discriminator(inputs ,output_size, sig = False, is_train=True, reuse=False, batch_size=24, output_units= 1):
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

def voxel2points(voxels, threshold=.3):
	l, m, n = voxels.shape
	X = []
	Y = []
	Z = []
	positions = np.where(voxels > threshold) # recieves position of all voxels
	offpositions = np.where(voxels < threshold) # recieves position of all voxels
	voxels[positions] = 1 # sets all voxels values to 1 
	voxels[offpositions] = 0 
	
	for i,j,k in zip(*positions):
		if np.sum(voxels[i-1:i+2,j-1:j+2,k-1:k+2])< 27 : #identifies if current voxels has an exposed face 
			X.append(i)
			Y.append(k)
			Z.append(j)
	
	return np.array(X),np.array(Y),np.array(Z)

def voxel2graph(filename, pred, epoch, threshold=.3):
	X,Y,Z = voxel2points(pred, threshold)
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.scatter(X, Y, Z, c=Z, cmap=cm.viridis, s=25, marker='.')
	plt.title('64-3D-RaSGAN [Epoch=%i]' % (epoch))
	plt.savefig(filename, bbox_inches='tight')
	plt.close('all')

def make_inputs_raw(file_batch):
	dt = np.dtype((np.uint8, (64,64,64)))
	models = [np.fromfile(f,dtype=dt).reshape((64,64,64)) for f in file_batch]
	models = np.array(models)
	models = models.astype(np.float32)
	start_time = time.time()
	return models, start_time
			
def generate_random_normal():
	sample = np.random.normal(0,1.0,[batchSize,200])
	sample = np.array(sample)
	sample = sample.astype(np.float32)
	return sample
	
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
	global batch_index
	batch_index += 1
	if(batch_index >= batchSize):
		batch_index = 0
	#save only one from batch per epoch to save space
	np.save(save_dir+str(epock)  , models[batch_index])
	
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
	
def render_graphs(save_dir,epoch, track_g_loss, track_d_loss, epoch_arr, track_d_validation_loss): 
	if not os.path.exists(save_dir+'/plots/'):
		os.makedirs(save_dir+'/plots/')
	if len(track_d_loss)> 51: 
		high_d = np.percentile(track_d_loss, 99)
		high_g = np.percentile(track_g_loss, 99)
		high_y = max([high_d, high_g]) * 1.6
		smoothed_d_loss = savitzky_golay(track_d_loss, 51, 3)
		smoothed_g_loss = savitzky_golay(track_g_loss, 51, 3)
		plt.plot(epoch_arr, track_d_loss, color='cornflowerblue', alpha=0.5)
		plt.plot(epoch_arr, smoothed_d_loss, color = 'navy', alpha=0.5)
		
		if(len(track_d_validation_loss)>1):
			smoothed_d_validation_loss = savitzky_golay(track_d_validation_loss, 51, 3)
			plt.plot(epoch_arr, track_d_validation_loss, color='lightgreen', alpha=0.5)
			plt.plot(epoch_arr, smoothed_d_validation_loss, color = 'limegreen', alpha=0.5)
			
		plt.plot(epoch_arr, track_g_loss, color='indianred', alpha=0.5)
		plt.plot(epoch_arr, smoothed_g_loss, color = 'crimson', alpha=0.5)
		if(len(track_d_validation_loss)>1):
			plt.legend(('Discriminator\'s loss','D-loss (Savitzky–Golay)','Discriminator\'s loss - validation set','D-loss - validation set (Savitzky–Golay)','Generator\'s loss', 'G-loss (Savitzky–Golay)'), loc='upper right')
		else:
			plt.legend(('Discriminator\'s loss','D-loss (Savitzky–Golay)','Generator\'s loss', 'G-loss (Savitzky–Golay)'), loc='upper right')
		
		plt.title('64-3D-RaSGAN [lrG=%.5f, lrD=%.5f]' % (args.generator_learning_rate, args.discriminator_learning_rate))
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.ylim([0,high_y])
		plt.grid(True)
		plt.savefig(save_dir+'/plots/' + str(epoch)+'.png' )
		plt.clf()

def save_values(save_dir,track_g_loss, track_d_loss, epoch_arr,track_d_validation_loss):
	np.save(save_dir+'/plots/track_g_loss', track_g_loss)
	np.save(save_dir+'/plots/track_d_loss', track_d_loss)
	np.save(save_dir+'/plots/epochs', epoch_arr)
	np.save(save_dir+'/plots/track_d_validation_loss', track_d_validation_loss)
	
def load_values(save_dir):
	outputs = []
	outputs.append(list(np.load(save_dir+'/plots/track_g_loss.npy')))
	outputs.append(list(np.load(save_dir+'/plots/track_d_loss.npy')))
	outputs.append(list(np.load(save_dir+'/plots/epochs.npy')))
	outputs.append(list(np.load(save_dir+'/plots/track_d_validation_loss.npy')))
	return outputs

###########################################
######### make directories ################
###########################################

if not os.path.exists(checkpoint_dir):
	os.makedirs(checkpoint_dir)

if not os.path.exists(save_dir):
	os.makedirs(save_dir)
	
if not os.path.exists(graph_3d_dir):
	os.makedirs(graph_3d_dir)

#######################################
########### inputs  ###################
#######################################

z = tf.random_normal((batchSize, 200), 0, 1)
#z_batch = tf.placeholder(tf.float32, [batchSize, 200] , name='z_batch')
#random_normal = generate_random_normal()
real_models = tf.placeholder(tf.float32, [batchSize, output_size, output_size, output_size] , name='real_models')
	
#######################################################
########## network computations #######################
#######################################################

#used for training generator
net_g, G_train =  generator_64(z, batch_size=batchSize, sig= False, is_train=True, reuse = False)

net_d , D_fake      = discriminator(G_train, output_size, batch_size= batchSize, sig = False, is_train = True, reuse = False)
net_d2, D_legit     = discriminator(real_models,  output_size, batch_size= batchSize, sig = False, is_train= True, reuse = True)
net_d2, D_eval      = discriminator(real_models,  output_size, batch_size= batchSize, sig = False, is_train= False, reuse = True) # this is for deciding whether to train the discriminator

##########################################
########### Loss calculations ############
##########################################

#Get logits
#logits_d1 = tf.subtract(D_legit, tf.reduce_mean(D_fake))
#logits_d2 = tf.subtract(D_fake, tf.reduce_mean(D_legit))
#logits_d21 = tf.subtract(D_eval, tf.reduce_mean(D_fake))
#logits_d22 = tf.subtract(D_fake, tf.reduce_mean(D_eval))
#logits_g1 = tf.subtract(D_legit, tf.reduce_mean(D_fake))
#logits_g2 = tf.subtract(D_fake, tf.reduce_mean(D_legit))

#d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_d1, labels=tf.ones_like(logits_d1)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_d2, labels=tf.zeros_like(logits_d2))) / 2.0
#d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_d21, labels=tf.ones_like(logits_d21)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_d22, labels=tf.zeros_like(logits_d22))) / 2.0
#g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_g1, labels=tf.zeros_like(logits_g1)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_g2, labels=tf.ones_like(logits_g2))) / 2.0

D_r_tilde = tf.nn.sigmoid(D_legit - tf.reduce_mean(D_fake))
D_f_tilde = tf.nn.sigmoid(D_fake - tf.reduce_mean(D_legit))
#eval
D_re_tilde = tf.nn.sigmoid(D_eval - tf.reduce_mean(D_fake))
D_fe_tilde = tf.nn.sigmoid(D_fake - tf.reduce_mean(D_eval))

d_loss = - tf.reduce_mean(tf.log(D_r_tilde + epsilon)) - tf.reduce_mean(tf.log(1 - D_f_tilde + epsilon))
d_loss2 = - tf.reduce_mean(tf.log(D_re_tilde + epsilon)) - tf.reduce_mean(tf.log(1 - D_fe_tilde + epsilon))
g_loss = - tf.reduce_mean(tf.log(D_f_tilde + epsilon)) - tf.reduce_mean(tf.log(1 - D_r_tilde + epsilon))

##########################################
############## Optimization ##############
##########################################

g_vars = tl.layers.get_variables_with_name('gen', True, True)   
d_vars = tl.layers.get_variables_with_name('dis', True, True)

d_optim = tf.train.AdamOptimizer(args.discriminator_learning_rate, beta1=0.5).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(args.generator_learning_rate, beta1=0.5).minimize(g_loss, var_list=g_vars)

#######################################
############# Training ################
#######################################

#Allocate only what is needed of gpu memory and then grow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
#sess=tf.Session()
sess.run(tf.global_variables_initializer())

#Debugging
tl.layers.print_all_variables()
print('G-vars count= ' + str(len(g_vars)))
print('D-vars count= ' + str(len(d_vars)))

# load checkpoints
if args.load: 
	load_networks(checkpoint_dir, sess, net_g, net_d, epoch = args.load_epoch)
	track_g_loss, track_d_loss, epoch_arr, track_d_validation_loss = load_values(save_dir)
else:     
	track_g_loss, track_d_loss, epoch_arr, track_d_validation_loss = [],[],[],[]

#Get training model files
files = glob.glob(args.data + '/*.raw')
files2 = files.copy()
#Get validation model files
files_validation = []
if(len(args.validation_data)>1):
	files_validation = glob.glob(args.data + '/*.raw')
	
if(len(files) == 0):
	print('Could not find any raw voxel grid files in ' + args.data)
	print('Please check that the .off files are converted with convert_data.py')
	exit(1)

if (len(files) < batchSize):
	print('Batch size is larger than sample pool size. No worries, adjusting batch size to ' + str(len(files)))
	batchSize = len(files)
	
Train_Dis = True 

if len(args.load_epoch)>1: 
	start = int(args.load_epoch)
else: 
	start = 0

errD_smooth = 0.0
errG_smooth = 0.0
alpha = 0.3
d_train_counter = 0
last_d_train_freq = 1
batch_index = 0

#trim data for graphs
if args.load:
	trim_count = int(args.load_epoch) * int(len(files)/batchSize)
	track_d_loss = track_d_loss[0:trim_count]
	track_g_loss = track_g_loss[0:trim_count]
	epoch_arr = epoch_arr[0:trim_count]
	if(len(track_d_validation_loss)>1):
		track_d_validation_loss = track_d_validation_loss[0:trim_count]

for epoch in range(start, args.epochs):
	random.shuffle(files)
	random.shuffle(files2)
	for idx in range(0, int(len(files)/batchSize)):
		file_batch = files[idx*batchSize:(idx+1)*batchSize]
		models, start_time = make_inputs_raw(file_batch)
		random_normal = generate_random_normal()
		
		#training the discriminator 
		if Train_Dis: 
			errD,_= sess.run([d_loss, d_optim] ,feed_dict={real_models: models})
			last_d_train_freq = d_train_counter
			d_train_counter = 0
		else: 
			returnArr = sess.run([d_loss2] ,feed_dict={real_models: models})
			errD = returnArr[0]
		
		#Validation loss
		if(len(files_validation) > batchSize):
			random.shuffle(files_validation)
			file_batch = files_validation[0:batchSize]
			models,_ = make_inputs_raw(file_batch)
			validation_returnArr = sess.run([d_loss2] ,feed_dict={real_models: models})
			track_d_validation_loss.append(validation_returnArr[0])
			
		
		#Resample again from the real data
		file_batch = files2[idx*batchSize:(idx+1)*batchSize]
		models,_ = make_inputs_raw(file_batch)
		
		#train generator
		errG,_,objects = sess.run([g_loss, g_optim, G_train], feed_dict={real_models: models})
		
		#Train discriminator on the next round?
		errG_smooth = (alpha * errG) + ((1.0 - alpha) * errG_smooth)
		errD_smooth = (alpha * errD) + ((1.0 - alpha) * errD_smooth)
		Train_Dis = ((2 * errD_smooth) > errG_smooth)
		d_train_counter += 1
		
		#Append arrays
		track_g_loss.append(errG)
		track_d_loss.append(errD)
		epoch_arr.append(epoch)
		
		print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, TrainD: %s, G/D train ratio: %i" \
					% (epoch, args.epochs, idx, int(len(files)/batchSize) , time.time() - start_time, errD, errG, Train_Dis, last_d_train_freq))
	#saving the model 
	if np.mod(epoch, args.save) == 0:
		save_networks(checkpoint_dir,sess, net_g, net_d, epoch)
	#saving generated objects
	if np.mod(epoch, args.sample ) == 0:     
		save_voxels(save_dir,objects, epoch )
	#saving learning info 
	if np.mod(epoch, args.graph) == 0: 
		render_graphs(save_dir,epoch, track_g_loss, track_d_loss, epoch_arr, track_d_validation_loss) #this will only work after a 50 iterations to allow for proper averaging 
		save_values(save_dir, track_g_loss, track_d_loss, epoch_arr, track_d_validation_loss)
	if np.mod(epoch, args.graph3d) == 0:
		newFile = graph_3d_dir + str(epoch) + '.png'
		voxel2graph(newFile, objects[batch_index], epoch)
		



    
