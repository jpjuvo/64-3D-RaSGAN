import numpy as np
import sys 
import os 
from subprocess import call
import glob
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


parser = argparse.ArgumentParser(description='Convert voxel files to 3D graph')
parser.add_argument('-f', '--file', help='File path. Convert single *.npy model file to a 3D scatter plot graph', type=str)
parser.add_argument('-n', '--name', help='Training run name for saving images from all  model files in that run', type=str)
args = parser.parse_args()

dirRun = True
workDir = ''
if not args.name:
	if not args.file:
		print('Please specify a run name with -n or a single file\'s path with -f')
		exit(1)
	else:
		dirRun = False
else:
	workDir = 'savepoint/' +  args.name + '/'
	if not os.path.exists(workDir):
		print('Could not find ' + workDir)
		exit(1)

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
	#trim based on coordinates
	#skim off parts from the left and back
        if(i > 44) or (k > 44):
           continue
        if np.sum(voxels[i-1:i+2,j-1:j+2,k-1:k+2])< 27 : #identifies if current voxels has an exposed face 
            X.append(i)
            Y.append(k)
            Z.append(j)
  
    return np.array(X),np.array(Y),np.array(Z)

def voxel2graph(filename, pred, threshold=.3):
	X,Y,Z = voxel2points(pred, threshold )
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	#ax.plot_trisurf(X,Y,Z, linewidth=0.2, antialiased=True)
	ax.scatter(X, Y, Z, c=Z, cmap=cm.copper, s=25, marker='.')
	if dirRun:
		plt.savefig(filename, bbox_inches='tight')
	else:
		plt.savefig(filename, bbox_inches='tight')
		plt.show()
		
	
	

modelFiles = ['']
if dirRun:
	modelFiles = glob.glob(workDir + '*.npy')
	print('found ' + str(len(modelFiles)) + ' models')
else:
	modelFiles = [args.file]
	
#make save directory
modelDir = workDir + '3D_graphs/'
if not os.path.exists(modelDir):
        os.makedirs(modelDir)
		
for mFile in tqdm(modelFiles):
    #Check if this has been done previously
    elts = os.path.split(mFile)
    nameonly = os.path.splitext(elts[1])
    newFile = modelDir + nameonly[0] + '.png'
    if((os.path.isfile(newFile)) and dirRun):
        continue
    #Load model
    models = np.load(mFile)
    #batch of models
    if len(models.shape) > 3:
        for i,m in enumerate(models):
            voxel2graph(newFile, m)
            #Load only first
            break
    else:	#single model
        voxel2graph(newFile, models)
		
if dirRun:
	print('Complete. Graph images were saved to ' + modelDir)


