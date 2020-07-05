import os.path as osp
import numpy as np
import glob
import xml.etree.ElementTree as et
from xml.dom import minidom
import argparse


def addShape(root, name ):
    shape = et.SubElement(root, 'shape' )
    shape.set('id', '{0}_object'.format(name) )
    shape.set('type', 'obj' )
    stringF = et.SubElement(shape, 'string' )
    stringF.set('name', 'filename' )
    stringF.set('value', '{0}'.format(name) )

    bsdf = et.SubElement(shape, 'bsdf')
    bsdf.set('id', 'mat')
    bsdf.set('type', 'dielectric' )

    specularR = et.SubElement(bsdf, 'rgb')
    specularR.set('name', 'specularReflectance')
    specularR.set('value', '1.0 1.0 1.0')

    specularT = et.SubElement(bsdf, 'rgb')
    specularT.set('name', 'specularTransmittance')
    specularT.set('value', '1.0 1.0 1.0')

    intInd = et.SubElement(bsdf, 'float')
    intInd.set('name', 'intIOR')
    intInd.set('value', '%.4f' % intIOR )

    extInd = et.SubElement(bsdf, 'float')
    extInd.set('name', 'extIOR')
    extInd.set('value', '%.4f' % extIOR )

    return root

def transformToXml(root ):
    rstring = et.tostring(root, 'utf-8')
    pstring = minidom.parseString(rstring)
    xmlString = pstring.toprettyxml(indent="    ")
    xmlString= xmlString.split('\n')
    xmlString = [x for x in xmlString if len(x.strip()) != 0 ]
    xmlString = '\n'.join(xmlString )
    return xmlString

def addSensor(root, fovValue, imWidth, imHeight, sampleCount):
    camera = et.SubElement(root, 'sensor')
    camera.set('type', 'perspective')
    fov = et.SubElement(camera, 'float')
    fov.set('name', 'fov')
    fov.set('value', '%.4f' % (fovValue) )
    fovAxis = et.SubElement(camera, 'string')
    fovAxis.set('name', 'fovAxis')
    fovAxis.set('value', 'x')
    film = et.SubElement(camera, 'film')
    film.set('type', 'hdrfilm')
    width = et.SubElement(film, 'integer')
    width.set('name', 'width')
    width.set('value', '%d' % (imWidth) )
    height = et.SubElement(film, 'integer')
    height.set('name', 'height')
    height.set('value', '%d' % (imHeight) )
    sampler = et.SubElement(camera, 'sampler')
    sampler.set('type', 'adaptive')
    sampleNum = et.SubElement(sampler, 'integer')
    sampleNum.set('name', 'sampleCount')
    sampleNum.set('value', '%d' % (sampleCount) )
    return root

parser = argparse.ArgumentParser()
parser.add_argument('--imNum', type=int, default=75, help='the number of images')
parser.add_argument('--imWidth', type=int, default=480, help='image width')
parser.add_argument('--imHeight', type=int, default=360, help='image height')
parser.add_argument('--sampleCount', type=int, default=64, help='the number of samples')
parser.add_argument('--fov', type=float, default=63.4149, help='the field of view in x axis')
parser.add_argument('--dist', type=float, default=3.0, help='the distance from camera to the target point')
parser.add_argument('--mode', default='train')
parser.add_argument('--fileRoot', default='./Shapes/', help='path to the file root')
parser.add_argument('--rs', default = 0, type=int, help='the starting point')
parser.add_argument('--re', default = 10, type=int, help='the end point')
parser.add_argument('--intIOR', type=float, default=1.4723, help='the index of refraction of glass')
parser.add_argument('--extIOR', type=float, default=1.0003, help='the index of refraction of air')
opt = parser.parse_args()
print(opt )


imNum = opt.imNum
imHeight = opt.imHeight
imWidth = opt.imWidth
fovValue = opt.fov
sampleCount = opt.sampleCount
dist = opt.dist
rs = opt.rs
re = opt.re
fileRoot = opt.fileRoot
intIOR = opt.intIOR
extIOR = opt.extIOR
mode = opt.mode

shapes = glob.glob(osp.join(fileRoot, mode, 'Shape*' ) )
for n in range(rs, min(re, len(shapes) ) ):
    shapeRoot = osp.join(fileRoot, mode, 'Shape__%d' % n )
    print('%d/%d: %s' % (n, min(re, len(shapes) ), shapeRoot ) )

    if osp.isfile(osp.join(shapeRoot, 'depth.xml') ):
        print('Warning: %s already exists' % shapeRoot )
        continue

    # Create rendering file for Depth maps
    root = et.Element('scene')
    root.set('version', '0.5.0')
    integrator = et.SubElement(root, 'integrator')
    integrator.set('type', 'path')

    # Add shape
    addShape(root, 'object.obj' )

    # Add sensor
    root = addSensor(root, fovValue, imWidth, imHeight, sampleCount)

    # Output xml file
    xmlString = transformToXml(root )
    with open(osp.join(shapeRoot, 'depth.xml'), 'w') as xmlOut:
        xmlOut.write(xmlString )

    # Generate camera file
    target = np.array([0.0, 0.0, 0.0], dtype = np.float32 )

    originSeed = np.random.random( (imNum, 2) ).astype(np.float32 )
    phi = originSeed[:, 0:1] * np.pi * 2
    theta = np.arccos(2 * originSeed[:, 1:2] - 1 )

    origin = np.concatenate([
        np.sin(theta ) * np.cos(phi ),
        np.sin(theta ) * np.sin(phi ),
        np.cos(theta ) ], axis=1 )
    origin = origin * dist

    xaxis = np.zeros((imNum * 3), dtype=np.float32 )
    indexAxis = np.arange(0, imNum ) * 3 + np.argmin(np.abs(origin), axis=1 )
    xaxis[indexAxis ] = 1.0
    xaxis = xaxis.reshape(imNum, 3 )

    up = np.concatenate([
        xaxis[:, 1:2] * origin[:, 2:3] - xaxis[:, 2:3] * origin[:, 1:2],
        xaxis[:, 2:3] * origin[:, 0:1] - xaxis[:, 0:1] * origin[:, 2:3],
        xaxis[:, 0:1] * origin[:, 1:2] - xaxis[:, 1:2] * origin[:, 0:1]
        ], axis = 1 )
    up = up / np.sqrt(np.sum(up * up, axis = 1) )[:, np.newaxis ]

    with open(osp.join(shapeRoot, 'depthCam.txt'), 'w') as camOut:
        camOut.write('%d\n' % imNum )
        for n in range(0, imNum ):
            camOut.write('%.4f %.4f %.4f\n' % (origin[n, 0], origin[n, 1], origin[n, 2] ) )
            camOut.write('%.4f %.4f %.4f\n' % (target[0], target[1], target[2] ) )
            camOut.write('%.4f %.4f %.4f\n' % (up[n, 0], up[n, 1], up[n, 2] ) )

    # Generate log file
    with open(osp.join(shapeRoot, 'depthLogPoint.txt'), 'w') as logOut:
        for n in range(0, imNum ):
            logOut.write('%d %d %d\n' % (0, 0, n+1) )

            y = up[n, :]
            z = origin[n, :]
            x = np.cross(y, z)

            y = y / np.sqrt(np.sum(y*y ) )
            z = z / np.sqrt(np.sum(z*z ) )
            x = x / np.sqrt(np.sum(x*x ) )
            rot = np.concatenate([x[:, np.newaxis], \
                    y[:, np.newaxis], z[:, np.newaxis] ], axis=1 )

            logOut.write('%.6f %.6f %.6f %.6f\n' % (rot[0, 0], rot[0, 1], rot[0, 2], origin[n, 0]) )
            logOut.write('%.6f %.6f %.6f %.6f\n' % (rot[1, 0], rot[1, 1], rot[1, 2], origin[n, 1]) )
            logOut.write('%.6f %.6f %.6f %.6f\n' % (rot[2, 0], rot[2, 1], rot[2, 2], origin[n, 2]) )
            logOut.write('%.6f %.6f %.6f %.6f\n' % (0, 0, 0, 1) )
