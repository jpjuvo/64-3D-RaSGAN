import sys
import os
import glob
import argparse
import subprocess
from tqdm import tqdm

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir,'../'))

parser = argparse.ArgumentParser(description='Convert ModelNet\'s .off files to a 3D voxel grid. Requires Patrick Min\'s Binvox software (http://www.patrickmin.com/binvox/)')
parser.add_argument('-m', '--model_dir', help='.off models directory (default=[base_dir]/ModelNet10/chair)', default=os.path.join(base_dir,'ModelNet10/chair'))
parser.add_argument('-b', '--binvox_binary', help='Full path to the Patrick Min\'s Binvox binary (default=[base_dir]/Binvox.exe)', default=os.path.join(base_dir,'Binvox.exe'))
args = parser.parse_args()

search_dir = args.model_dir
binvox = args.binvox_binary
voxels = 64

#Check that the binvox binary is found
if not os.path.isfile(binvox):
	print('Error: Could not locate Binvox binary at: ' + binvox)
	print('Please specify the path of your binvox binary with: -b [binvox_binary])')
	print('You can download the binary for your os from http://www.patrickmin.com/binvox/')
	exit(1)

#Check search directory
if not os.path.isdir(search_dir):
	print('Error: Could not find ' + search_dir)
	exit(1)
	
print('-------------------- ARGUMENTS --------------------')
print('Search directory selected = ' + search_dir)
print('Patrick Min\'s Binvox binary path = ' + binvox)
print('---------------------------------------------------')

counter = 0
prev_conv_counter = 0
#supress output of the binvox to stdout
FNULL = open(os.devnull, 'w')
binvox_args = binvox + ' -d ' + str(voxels) + ' -t raw '

#Go through all .off model files recursively in the specified path
for (dirpath, dirnames, filenames) in os.walk(search_dir):
	if(len(filenames) == 0):
		continue
	print('Converting files in: ' + dirpath + '...')
	for filename in tqdm(filenames):
		if filename.endswith('.off'):
			#Parse full path
			fname = os.path.join(dirpath,filename)
			parts = fname.split('.')
			rawfile = parts[0] + '.raw'

			#Check if this file has already been converted
			if os.path.isfile(rawfile):
				prev_conv_counter += 1
				continue
			
			#Convert
			subprocess.call(binvox_args + fname, stdout=FNULL,shell=False)
			
			#Count converted files
			counter += 1
		else:
			continue

	print(dirpath + ' done')
print(str(counter) + ' .off files converted to raw 3D voxel grid.')

if (prev_conv_counter > 0):
	print(str(prev_conv_counter) + ' .off files were already converted.')