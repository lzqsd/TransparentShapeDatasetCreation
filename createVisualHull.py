import numpy as np
import open3d as o3d
import cv2
from skimage import measure
import os
import argparse
import scipy.ndimage as ndimage
import os.path as osp
import glob
import h5py
import trimesh as trm

parser = argparse.ArgumentParser()
parser.add_argument('--imWidth', type=int, default=480, help='image width')
parser.add_argument('--imHeight', type=int, default=360, help='image height')
parser.add_argument('--resolution', type=int, default=128, help='resolution of the visual hull')
parser.add_argument('--fov', type=float, default=63.4149, help='the field of view in x axis')
parser.add_argument('--mode', default='train')
parser.add_argument('--fileRoot', default='./Shapes/', help='path to the file root')
parser.add_argument('--imageRoot', default='./Images', help='path to the image root')
parser.add_argument('--camNum', type=int, required=True, help='Number of cameras')
parser.add_argument('--rs', default = 0, type=int, help='the starting point')
parser.add_argument('--re', default = 10, type=int, help='the end point')
parser.add_argument('--forceOutput', action='store_true', help='Overwrite old results')
opt = parser.parse_args()
print(opt )


fileRoot = osp.join(opt.fileRoot, opt.mode )
imageRoot = osp.join(opt.imageRoot + '%d' % opt.camNum, opt.mode )
resolution = opt.resolution
imHeight, imWidth = opt.imHeight, opt.imWidth
fov = opt.fov / 180.0 * np.pi
fovX = np.tan(fov / 2.0 )
pixelSize = 0.5 * imWidth / fovX
rs = opt.rs
re = opt.re

shapes = glob.glob(osp.join(fileRoot, 'Shape*' ) )
for shapeId in range(rs, min(re, len(shapes) ) ):

    shapeRoot = osp.join(fileRoot, 'Shape__%d' % shapeId )
    print('%d/%d: %s' % (shapeId, min(re, len(shapes ) ), shapeRoot ) )

    if osp.isfile(osp.join(shapeRoot, 'visualHullSubd_%d.ply' % (opt.camNum ) ) ):
        print('Warning: visual hull of %s has already been created' % shapeRoot )
        if not opt.forceOutput:
            continue

    minX, maxX = -1, 1
    minY, maxY = -1, 1
    minZ, maxZ = -1, 1

    y, x, z = np.meshgrid(
            np.linspace(minX, maxX, resolution ),
            np.linspace(minY, maxY, resolution ),
            np.linspace(minZ, maxZ, resolution ) )
    x = x[:, :, :, np.newaxis ]
    y = y[:, :, :, np.newaxis ]
    z = z[:, :, :, np.newaxis ]
    coord = np.concatenate([x, y, z], axis=3 )
    volume = -np.ones(x.shape ).squeeze()

    camRot = []
    camTrans = []
    with open(osp.join(shapeRoot, 'cam%d.txt' % opt.camNum ), 'r') as camIn:
        camNum = int(camIn.readline().strip() )
        camNum = opt.camNum
        print('Number of Cameras: %d' % camNum )
        for camId in range(0, camNum ):
            originStr = camIn.readline().strip()
            targetStr = camIn.readline().strip()
            upStr = camIn.readline().strip()

            origin = originStr.split(' ')
            origin = np.array([float(x) for x in origin ] )

            target = targetStr.split(' ')
            target = np.array([float(x) for x in target ] )

            up = upStr.split(' ')
            up = np.array([float(x) for x in up ] )

            camTrans.append(origin )

            yAxis = up / np.sqrt(np.sum(up * up ) )
            zAxis = target - origin
            zAxis = zAxis / np.sqrt(np.sum(zAxis * zAxis ) )
            assert( np.sum(yAxis * zAxis) < 1e-3 )

            xAxis = np.cross(zAxis, yAxis )
            xAxis = xAxis / np.sqrt(np.sum(xAxis * xAxis ) )

            xAxis = xAxis[np.newaxis, :]
            yAxis = yAxis[np.newaxis, :]
            zAxis = zAxis[np.newaxis, :]

            rot = np.concatenate([xAxis, yAxis, zAxis ], axis=0 )
            camRot.append(rot )

    # Start to build the voxel
    meshName = osp.join(shapeRoot, 'visualHull_%d.ply' % (opt.camNum ) )
    if not osp.isfile(meshName ):
        for n in range(0, camNum ):
            print('Process %d/%d' % (n, camNum ) )

            # Build mask
            fileName = osp.join(osp.join(imageRoot, 'Shape__%d' % shapeId),
                    'imtwoBounce_%d.h5') % (n+1)
            hf = h5py.File(fileName, 'r')
            twoBounce = hf.get('data')

            seg = twoBounce[:, :, 6]
            mask =seg.reshape(imWidth * imHeight )

            rot = camRot[n ]
            trans = camTrans[n ]
            rot = rot.reshape([1, 1, 1, 3, 3] )
            trans = trans.reshape([1, 1, 1, 3] )

            coordCam = coord -  trans
            xCam = np.sum(coordCam * rot[:, :, :, 0, :], axis = 3)
            yCam = np.sum(coordCam * rot[:, :, :, 1, :], axis = 3)
            zCam = np.sum(coordCam * rot[:, :, :, 2, :], axis = 3)
            assert(np.all(zCam > 0 )  )

            xCam = xCam / zCam
            yCam = yCam / zCam

            xId = xCam * pixelSize + imWidth / 2.0
            yId = -yCam * pixelSize + imHeight / 2.0

            xInd = np.logical_and(xId >= 0, xId < imWidth-0.5 )
            yInd = np.logical_and(yId >= 0, yId < imHeight-0.5 )
            imInd = np.logical_and(xInd, yInd )

            xImId = np.round(xId[imInd ] ).astype(np.int32 )
            yImId = np.round(yId[imInd ] ).astype(np.int32 )

            maskInd = mask[yImId * imWidth + xImId ]

            volumeInd = imInd.copy()
            volumeInd[imInd == 1] = maskInd

            volume[volumeInd == 0] = 1

            print('Occupied voxel: %d' % np.sum( (volume > 0).astype(np.float32 ) ) )

        verts, faces, normals, _ = measure.marching_cubes_lewiner(volume, 0)
        print('Vertices Num: %d' % verts.shape[0] )
        print('Normals Num: %d' % normals.shape[0] )
        print('Faces Num: %d' % faces.shape[0] )

        axisLen = float(opt.resolution -1 ) / 2.0
        verts = (verts - axisLen ) / axisLen
        mesh = trm.Trimesh(vertices = verts, vertex_normals = normals, faces = faces)

        mesh.export(osp.join(shapeRoot, 'visualHull_%d.ply' % (opt.camNum ) ) )

    cmd = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i %s -o %s -om vn -s remeshVisualHull.mlx' % \
            (osp.join(shapeRoot, 'visualHull_%d.ply' % opt.camNum ), osp.join(shapeRoot, 'visualHullSubd_%d.ply' % opt.camNum ) )
    os.system(cmd )
