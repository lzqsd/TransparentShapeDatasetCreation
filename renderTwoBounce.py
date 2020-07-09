import os.path as osp
import numpy as np
import glob
import struct
import cv2
import scipy.ndimage as ndimage
import os
import argparse
import struct
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train')
parser.add_argument('--camNum', type=int, required=True )
parser.add_argument('--renderProgram', default='./OptixRenderer/src/bin/optixRenderer', help='path to the rendering program')
parser.add_argument('--fileRoot', default='./Shapes/', help='path to the file root')
parser.add_argument('--forceOutput', action='store_true', help='whether to overwrite previous results')
parser.add_argument('--outputRoot', default='Images', help='path to the output root')
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
        files = glob.glob(osp.join(outputDir, 'imtwoBounce_*.h5') )
        if len(files ) != 0:
            if not opt.forceOutput:
                continue

    output = osp.join('../../../', outputDir, 'im.rgbe')

    xmlFile = osp.join(shapeRoot, 'im.xml')
    camFile = 'cam%d.txt' % opt.camNum

    if opt.forceOutput:
        cmd1 = '%s -f %s -o %s -c %s -m 7 --forceOutput' % (renderProgram, xmlFile, output, camFile )
    else:
        cmd1 = '%s -f %s -o %s -c %s -m 7' % (renderProgram, xmlFile, output, camFile )

    os.system(cmd1 )

    results = glob.glob(osp.join(outputDir, 'imtwoBounce_*.dat') )
    camNum = len(results )
    cnt = 0
    for resName in results:
        cnt += 1
        print('Compressing %d/%d: %s' % (cnt, camNum, resName ) )
        with open(resName, 'rb') as f:
            byte = f.read(4)
            height = struct.unpack('i', byte )[0]
            byte = f.read(4)
            width = struct.unpack('i', byte )[0]

            byte = f.read()
            twoBounce = np.array(struct.unpack(
                str(width * height * 14) + 'f', byte ), dtype=np.float32 )
            twoBounce = twoBounce.reshape([height, width, 14])

            h5Name = resName.replace('.dat', '.h5')
            with h5py.File(h5Name, 'w') as fOut:
                fOut.create_dataset('data', data = twoBounce, compression='lzf')
        os.system('rm %s' % resName )



