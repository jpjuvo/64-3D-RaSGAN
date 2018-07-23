import wget
import os
import sys
from pathlib import Path
import zipfile

def continue_download():
	while True:
		reply = str(input('The ModelNet10.zip file is over 450 MB. Proceed to download (y/n): ')).lower().strip()
		if reply[0] == 'y':
			return True
		if reply[0] == 'n':
			return False
		else:
			print('please reply with y or n')

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir,'../'))

url = 'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'

download_path = base_dir + 'ModelNet10.zip'
final_path = base_dir + 'ModelNet10'

if os.path.isdir(final_path):
	print(final_path + ' directory already exists.')
	exit(0)

#The file is quite large so ask the user before downloading
if not continue_download():
	exit(0)
	
if not os.path.isfile(download_path):
	print('Beginning to download Princeton ModelNet10...')
	print('url: ' + url)
	wget.download(url, download_path)
	print('Download complete.')

#Check that the zipfile is found
if not os.path.exists(download_path):
	print('Error: Could not locate ModelNet10.zip')
	exit(1)

print('Unzipping ModelNet10...')

with zipfile.ZipFile(download_path , 'r') as zip_ref:
	zip_ref.extractall(base_dir)

print('Unzip complete, removing zip file...')
os.remove(download_path)

print('Complete')
