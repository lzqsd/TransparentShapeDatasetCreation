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
parser.add_argument('--imNum', type=int, default=75, help='the number of images')
parser.add_argument('--imWidth', type=int, default=480, help='image width')
parser.add_argument('--imHeight', type=int, default=360, help='image height')
parser.add_argument('--fov', type=float, default=63.4149, help='the field of view in x axis')
parser.add_argument('--mode', default='train')
parser.add_argument('--renderProgram', default='/home/zhl/OptixRenderer/src/bin/optixRenderer', help='path to the rendering program')
parser.add_argument('--fileRoot', default='./Shapes/', help='path to the file root')
parser.add_argument('--rs', default = 0, type=int, help='the starting point')
parser.add_argument('--re', default = 10, type=int, help='the end point')
opt = parser.parse_args()
print(opt )

imNum = opt.imNum
imHeight = opt.imHeight
imWidth = opt.imWidth
fovValue = opt.fov
rs = opt.rs
re = opt.re
fileRoot = opt.fileRoot
renderProgram = opt.renderProgram
mode = opt.mode

erodeBorder = 3
colmapTrim = 7
colmapDepth = 8
colmapNumThreads = 8

fovX = np.tan(fovValue / 360.0 * np.pi )
fovY = fovX / float(imWidth) * float(imHeight)

shapes = glob.glob(osp.join(fileRoot, mode, 'Shape*') )
for n in range(rs, min(re, len(shapes)) ):
    shapeRoot = osp.join(fileRoot, mode, 'Shape__%d' % n )
    print('%d/%d: %s' % (n, min(re, len(shapes) ), shapeRoot ) )

    if osp.isfile(osp.join(shapeRoot, 'poissonSubd.ply') ):
        print('Warning: %s already exists.' % shapeRoot )
        continue

    if not osp.isfile(osp.join(shapeRoot, 'poisson.ply') ):
        xmlFile = osp.join(shapeRoot, 'depth.xml')
        camFile = 'depthCam.txt'
        cmd1 = '%s -f %s -o im.png -c %s -m 5 --forceOutput' % (renderProgram, xmlFile, camFile )
        cmd2 = '%s -f %s -o im.png -c %s -m 2 --forceOutput' % (renderProgram, xmlFile, camFile )

        os.system(cmd1 )
        os.system(cmd2 )

        logFile = osp.join(shapeRoot, 'depthLogPoint.txt' )
        poseArray = []
        with open(logFile, 'r')  as logIn:
            for n in range(0, imNum ):
                metaData = logIn.readline()

                mat = []
                for m in range(0, 4):
                    arr = logIn.readline().split(' ')
                    arr = np.array([float(x) for x in arr ]  )[np.newaxis, :]
                    mat.append(arr )
                mat = np.concatenate(mat, axis=0 )
                poseArray.append(mat )

        normalArr = []
        pointArr = []
        for n in range(0, imNum ):
            depthName = osp.join(shapeRoot, 'imdepth_%d.dat' % (n+1 ) )
            normalName = osp.join(shapeRoot, 'imnormal_%d.png' % (n+1) )

            with open(depthName, 'rb') as depthIn:
                hBuffer = depthIn.read(4 )
                height = struct.unpack('i', hBuffer )[0]
                wBuffer = depthIn.read(4 )
                width = struct.unpack('i', wBuffer )[0]
                dBuffer = depthIn.read(4 * width * height )
                depth = np.asarray(struct.unpack('f'*height*width, dBuffer), dtype=np.float32 )
                depth = depth.reshape([height, width ] )

            mask = (depth > 0.1)
            mask = ndimage.binary_erosion(mask, structure = np.ones((erodeBorder, erodeBorder) ) )
            mask = mask.astype(np.float32 )
            maskNormal = mask[:, :, np.newaxis ]
            maskNormal = np.concatenate([maskNormal, maskNormal, maskNormal], axis=2)

            depth = depth * mask

            normalIm = cv2.imread(normalName )[:, :, ::-1]
            normalIm = normalIm.astype(np.float32 )
            normalIm = (normalIm - 127.5) / 127.5
            normalIm = normalIm / np.sqrt(np.maximum(np.sum(normalIm * normalIm, axis=2), 1e-10) )[:, :, np.newaxis]

            depthValues = depth[mask == 1]
            pointNum = depthValues.shape[0]

            normalValues = normalIm[maskNormal == 1]
            normalValues = normalValues.reshape([pointNum, 3] )

            x, y = np.meshgrid(np.linspace(-1, 1, 480), np.linspace(-1, 1, 360) )
            x = fovX * x
            y = -fovY * y
            z = -np.ones( (imHeight, imWidth ) )
            w = np.ones( (imHeight, imWidth) )

            x = x[:, :, np.newaxis ]
            y = y[:, :, np.newaxis ]
            z = z[:, :, np.newaxis ]
            w = w[:, :, np.newaxis ]
            point = np.concatenate([x, y, z, w], axis=2 )

            maskPoint = mask[:, :, np.newaxis ]
            maskPoint = np.concatenate([maskPoint, maskPoint, \
                    maskPoint, maskPoint], axis=2 )
            pointValues = point[maskPoint == 1 ]
            pointValues = pointValues.reshape([pointNum, 4])
            pointValues = pointValues * depthValues[:, np.newaxis]
            pointValues[:, 3] = 1.0

            posMat = poseArray[n]
            rotMat = posMat[0:3, 0:3]

            normalValues = normalValues.transpose([1, 0])
            pointValues = pointValues.transpose([1, 0])

            normalValues = np.matmul(rotMat, normalValues )
            pointValues = np.matmul(posMat, pointValues )

            normalArr.append(normalValues )
            pointArr.append(pointValues[0:3, :] )

        normalArr = np.concatenate(normalArr, axis=1 )
        pointArr = np.concatenate(pointArr, axis=1 )

        normalArr = normalArr.transpose([1, 0])
        pointArr = pointArr.transpose([1, 0])
        colorArr = (normalArr + 1) * 0.5

        shapeName = osp.join(shapeRoot, 'pointCloud.ply')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointArr )
        pcd.normals = o3d.utility.Vector3dVector(normalArr )
        pcd.colors = o3d.utility.Vector3dVector(colorArr )
        o3d.io.write_point_cloud(shapeName, pcd)

        cmd = 'colmap poisson_mesher --input_path %s ' \
                + '--PoissonMeshing.trim %d ' \
                + '--PoissonMeshing.depth %d ' \
                + '--PoissonMeshing.num_threads %d ' \
                + '--output_path %s'
        outputName = osp.join(shapeRoot, 'poisson.ply')
        cmd = cmd % (shapeName, colmapTrim, colmapDepth, colmapNumThreads, outputName )
        print(cmd )
        os.system(cmd )

        # Remove the depth and normal
        cmd1 = 'rm %s' % osp.join(shapeRoot, 'imnormal_*.png')
        cmd2 = 'rm %s' % osp.join(shapeRoot, 'imdepth_*.dat')
        cmd3 = 'rm %s' % osp.join(shapeRoot, 'pointCloud.ply')
        os.system(cmd1 )
        os.system(cmd2 )
        os.system(cmd3 )

    # Smooth the mesh
    cmd = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i %s -o %s -om vn -s remesh.mlx' % (
            osp.join(shapeRoot, 'poisson.ply'), osp.join(shapeRoot, 'poissonSubd.ply') )
    os.system(cmd )
