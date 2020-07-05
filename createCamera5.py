import os.path as osp
import numpy as np
import glob
import argparse
import random



parser = argparse.ArgumentParser()
parser.add_argument('--imNum', type=int, default=5, help='the number of camera view')
parser.add_argument('--dist', type=float, default=3.0, help='the distance from camera to the target point')
parser.add_argument('--mode', default='train')
parser.add_argument('--fileRoot', default='./Shapes/', help='path to the file root')
parser.add_argument('--rs', default = 0, type=int, help='the starting point')
parser.add_argument('--re', default = 10, type=int, help='the end point')
opt = parser.parse_args()
print(opt )

imNum = opt.imNum
dist = opt.dist
rs = opt.rs
re = opt.re
mode = opt.mode
fileRoot = opt.fileRoot

shapes = glob.glob(osp.join(fileRoot, mode, 'Shape*') )

thetaValue = np.array([90], dtype=np.float32 )
phiValue = np.array([0, 72, 144, 216, 288], dtype=np.float32 )
phiOrigin, thetaOrigin = np.meshgrid(phiValue, thetaValue )
phiOrigin, thetaOrigin = phiOrigin.flatten(), thetaOrigin.flatten()

thetaRandom = 10
phiRandom = 20

for n in range(rs, min(re, len(shapes) ) ):
    shapeRoot = osp.join(fileRoot, mode, 'Shape__%d' % n )
    print('%d/%d: %s' % (n, min(re, len(shapes) ), shapeRoot ) )

    if osp.isfile( osp.join(shapeRoot, 'cam5.txt' ) ):
        print("Warning: %s already exisits." % shapeRoot )
        continue

    # Generate camera file
    target = np.random.random(3)* 0.1 - 0.05

    theta = thetaOrigin + (np.random.random(imNum )  - 0.5) * 2 * thetaRandom
    phi = phiOrigin + (np.random.random(imNum) - 0.5) * 2 * phiRandom
    theta = (theta / 180.0 * np.pi)[:, np.newaxis ]
    phi = (phi / 180.0 * np.pi)[:, np.newaxis ]

    origin = np.concatenate([
        np.sin(theta ) * np.cos(phi ),
        np.cos(theta ),
        np.sin(theta ) * np.sin(phi ) ], axis=1 )
    origin = origin * dist

    up = np.zeros(origin.shape )
    up[:, 1] = 1
    up = up + (np.random.random(up.shape ) * 0.2 - 0.1)

    zaxis = target - origin
    zaxis = zaxis / np.sqrt( (np.sum(zaxis * zaxis, axis=1 )[:, np.newaxis ] ) )
    up = up - np.sum(up * zaxis, axis=1 )[:, np.newaxis ] * zaxis
    up = up / np.sqrt(np.sum(up * up, axis=1)[:, np.newaxis ] )

    with open(osp.join(shapeRoot, 'cam5.txt'), 'w') as camOut:
        camOut.write('%d\n' % imNum )
        for n in range(0, imNum ):
            camOut.write('%.4f %.4f %.4f\n' % (origin[n, 0], origin[n, 1], origin[n, 2] ) )
            camOut.write('%.4f %.4f %.4f\n' % (target[0], target[1], target[2] ) )
            camOut.write('%.4f %.4f %.4f\n' % (up[n, 0], up[n, 1], up[n, 2] ) )
