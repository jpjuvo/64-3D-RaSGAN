import numpy as np
import sys 
import os 
from subprocess import call
import glob
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Convert voxel files to mesh models (.obj)')
parser.add_argument('-n', '--name', help='Training run name for converting all  model files in that run', type=str)
parser.add_argument('-f', '--file', help='File path. Convert single *.npy model file to *.obj format', type=str)
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

# These notes are from https://github.com/EdwardSmith1884/3D-IWGAN:
# this is mostly from https://github.com/chrischoy/3D-R2N2/blob/master/lib/voxel.py 
# though I sped up the voxel2mesh function considerably, now only surface voxels are saved
# this is only really important for very large models 

def voxel2mesh(voxels, threshold=.3):
    cube_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                  [1, 1, 1]]  # 8 points

    cube_faces = [[0, 1, 2], [1, 3, 2], [2, 3, 6], [3, 7, 6], [0, 2, 6], [0, 6, 4], [0, 5, 1],
                  [0, 4, 5], [6, 7, 5], [6, 5, 4], [1, 7, 3], [1, 5, 7]]  # 12 face

    cube_verts = np.array(cube_verts)
    cube_faces = np.array(cube_faces) + 1

    l, m, n = voxels.shape

    scale = 0.01
    cube_dist_scale = 1.0 #1.1
    verts = []
    faces = []
    curr_vert = 0

    positions = np.where(voxels > threshold) # recieves position of all voxels
    offpositions = np.where(voxels < threshold) # recieves position of all voxels
    voxels[positions] = 1 # sets all voxels values to 1 
    voxels[offpositions] = 0 
	
    for i,j,k in zip(*positions):
	#trim based on coordinates
	#skim off parts from the left and back
        if(i > 64) or (k > 64) or (j > 64):
           continue
        if np.sum(voxels[i-1:i+2,j-1:j+2,k-1:k+2])< 27 : #identifies if current voxels has an exposed face 
            verts.extend(scale * (cube_verts + cube_dist_scale * np.array([[i, j, k]])))
            faces.extend(cube_faces + curr_vert)
            curr_vert += len(cube_verts)   
    return np.array(verts), np.array(faces)


def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))


def voxel2obj(filename, pred, threshold=.3):
    verts, faces = voxel2mesh(pred, threshold )
    write_obj(filename, verts, faces)

modelFiles = ['']
if dirRun:
	modelFiles = glob.glob(workDir + '*.npy')
	print('found ' + str(len(modelFiles)) + ' models')
else:
	modelFiles = [args.file]

#make save directory
modelDir = workDir + 'models/'
if not os.path.exists(modelDir):
        os.makedirs(modelDir)
		
for mFile in tqdm(modelFiles):
    #Check if this has been done previously
    elts = os.path.split(mFile)
    nameonly = os.path.splitext(elts[1])
    newFile = modelDir + nameonly[0] + '.obj'
    if((os.path.isfile(newFile)) and dirRun):
        continue
    #Load model
    models = np.load(mFile)
    #batch of models
    if len(models.shape) > 3:
        for i,m in enumerate(models):
            voxel2obj(newFile, m)
            #Load only first
            break
    else:	#single model
        voxel2obj(newFile, models)
		
if dirRun:
	print('Complete. Files were saved to ' + modelDir)
else:
	print('Complete. .obj File saved to /models/ folder.')


