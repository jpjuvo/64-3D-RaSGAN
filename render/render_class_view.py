#!/usr/bin/python

import os.path as osp
import sys
import argparse
import os, tempfile, glob, shutil

BASE_DIR = osp.dirname(__file__)
sys.path.append(osp.join(BASE_DIR,'../'))
from global_variables import *


parser = argparse.ArgumentParser(description='Render Model Images (Requires Blender)')
parser.add_argument('-m', '--model_file', help='CAD Model obj filename', default=osp.join(BASE_DIR,'sample_model/model.obj'))
parser.add_argument('-a', '--azimuth', default='50')
parser.add_argument('-e', '--elevation', default='30')
parser.add_argument('-tex', '--texture', default='', type=str)
parser.add_argument('-t', '--tilt', default='0')
parser.add_argument('-d', '--distance', default='2.0')
parser.add_argument('-o', '--output_img', help='Output img filename.', default=osp.join(BASE_DIR, 'demo_img.png')) 
parser.add_argument('-b','--blender_path', help='Path to Blender executable (Get Blender from https://www.blender.org/)', default='C:/Program Files/Blender Foundation/Blender/blender.exe')
args = parser.parse_args()

blank_file = osp.join(g_blank_blend_file_path)
render_code = osp.join(g_render4cnn_root_folder, 'render_pipeline/render_model_views.py')

# MK TEMP DIR
temp_dirname = tempfile.mkdtemp()
view_file = osp.join(temp_dirname, 'view.txt')
view_fout = open(view_file,'w')
view_fout.write(' '.join([args.azimuth, args.elevation, args.tilt, args.distance]))
view_fout.close()
blender_exe =args.blender_path

#Check that the Blender binary is found
if not os.path.isfile(blender_exe):
	print('Error: Could not locate Blender at: ' + blender_exe)
	print('Please specify the path of your Blender executable with: -b [blender_path])')
	print('You can download blender for your os from https://www.blender.org')
	exit(1)

#Check that the Model file is found
if not os.path.isfile(args.model_file):
	print('Error: Could not locate model file: ' + args.model_file)
	print('Please specify the path of your model file with: -m [model_file])')
	print('Example: -m C:/sample_model.obj')
	exit(1)
	
try:
    render_cmd = '\"%s\" %s --background --python %s %s %s %s %s %s %s' % (blender_exe, blank_file, render_code,args.texture, args.model_file, 'xxx', 'xxx', view_file, temp_dirname)
    print(render_cmd)
    os.system(render_cmd)
    imgs = glob.glob(temp_dirname+'/*.png')
    shutil.move(imgs[0], args.output_img.split('.')[-2]+'.png')
except:
    print('render failed. render_cmd: %s' % (render_cmd))

# CLEAN UP
shutil.rmtree(temp_dirname)
