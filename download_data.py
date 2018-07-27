import wget
import os
import sys
from pathlib import Path
import zipfile
import tarfile
import gzip
import shutil

def continue_download(is40 = False):
	queryStr = 'The ModelNet10.zip file is over 450 MB. Proceed to download (y/n): '
	if is40:
		queryStr =  'The ModelNet40.tar file is 2 GB and over 9 GB uncompressed. Proceed to download (y/n): '
	while True:
		reply = str(input(queryStr)).lower().strip()
		if reply[0] == 'y':
			return True
		if reply[0] == 'n':
			return False
		else:
			print('please reply with y or n')
			
def query_dataset():
	while True:
		reply = str(input('Choose dataset, ModelNet10 (1) or manually aligned subset of the ModelNet40 (2):')).lower().strip()
		if reply[0] == '1':
			return True
		if reply[0] == '2':
			return False
		else:
			print('please reply with 1 or 2')

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir,'../'))

modelnet10url = 'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
modelnet40url = 'https://lmb.informatik.uni-freiburg.de/resources/datasets/ORION/modelnet40_manually_aligned.tar'
modelnet40 = False

url = 'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
download_path = base_dir + 'ModelNet10.zip'
final_path = base_dir + 'ModelNet10'

if not query_dataset():
	#modelnet40
	modelnet40 = True
	url = modelnet40url
	download_path = base_dir +  'ModelNet40.tar'
	final_path = base_dir + 'ModelNet40'

if os.path.isdir(final_path):
	print(final_path + ' directory already exists.')
	exit(0)

#The file is quite large so ask the user before downloading
if not continue_download(modelnet40):
	exit(0)
	
if not os.path.isfile(download_path):
	print('Beginning to download Princeton ModelNet...')
	print('url: ' + url)
	wget.download(url, download_path)
	print('Download complete.')

#Check that the zipfile is found
if not os.path.exists(download_path):
	print('Error: Could not locate ' + download_path)
	exit(1)

print('Unzipping ModelNet...')

if not modelnet40:
	with zipfile.ZipFile(download_path , 'r') as zip_ref:
		zip_ref.extractall(base_dir)
else:
	print('Please wait. This may take a few minutes...')
	with gzip.open(download_path,'rb') as f:
		with open('temp.tar', 'wb') as f_out:
			shutil.copyfileobj(f, f_out)
	print('Unpacking contents...')
	tar = tarfile.open('temp.tar', "r:")
	tar.extractall(path=final_path)
	tar.close()
	os.remove('temp.tar')

print('Unzip complete, removing temporary files...')
os.remove(download_path)

print('Complete')
