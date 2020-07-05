import os.path as osp
import numpy as np
import glob
import struct
import cv2
import scipy.ndimage as ndimage
import open3d as o3d
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train')
parser.add_argument('--camNum', type=int, required = True )
parser.add_argument('--renderProgram', default='/home/zhl/OptixRenderer/src/bin/optixRenderer', help='path to the rendering program')
parser.add_argument('--fileRoot', default='./Shapes/', help='path to the file root')
parser.add_argument('--outputRoot', default='Images', help='path to the output root')
parser.add_argument('--forceOutput', action='store_true', help='Overwrite previous results')
parser.add_argument('--rs', default = 0, type=int, help='the starting point')
parser.add_argument('--re', default = 10, type=int, help='the end point')
opt = parser.parse_args()
print(opt )

rs = opt.rs
re = opt.re
mode = opt.mode
fileRoot = opt.fileRoot
outputRoot = opt.outputRoot + '%d' % opt.camNum
renderProgram = opt.renderProgram

shapes = glob.glob(osp.join(fileRoot, mode, 'Shape*') )
for n in range(rs, min(re, len(shapes)) ):
    shapeRoot = osp.join(fileRoot, mode, 'Shape__%d' % n)
    print('%d/%d: %s' % (n, min(re, len(shapes) ), shapeRoot ) )

    shapeId = shapeRoot.split('/')[-1]
    outputDir = osp.join(outputRoot, mode, shapeId )

    if not osp.isdir(outputDir ):
        os.system('mkdir -p %s' % outputDir )
    else:
        print('Warning: output directory %s already exists' % (outputDir ) )
        files = glob.glob(osp.join(outputDir, '*.rgbe') )
        if len(files ) == opt.camNum:
            print(len(files ) )
            continue

    output = osp.join('../../../', outputDir, 'im.rgbe')

    xmlFile = osp.join(shapeRoot, 'im.xml')
    camFile = 'cam%d.txt' % opt.camNum

    if opt.forceOutput:
        cmd1 = '%s -f %s -o %s -c %s -m 0 --forceOutput' % (renderProgram, xmlFile, output, camFile )
        #cmd2 = '%s -f %s -o %s -c %s -m 2 --forceOutput' % (renderProgram, xmlFile, output, camFile )
        #cmd3 = '%s -f %s -o %s -c %s -m 4 --forceOutput' % (renderProgram, xmlFile, output, camFile )
    else:
        cmd1 = '%s -f %s -o %s -c %s -m 0' % (renderProgram, xmlFile, output, camFile )
        #cmd2 = '%s -f %s -o %s -c %s -m 2' % (renderProgram, xmlFile, output, camFile )
        #cmd3 = '%s -f %s -o %s -c %s -m 4' % (renderProgram, xmlFile, output, camFile )

    os.system(cmd1 )
    #os.system(cmd2 )
    #os.system(cmd3 )
