import os.path as osp
import numpy as np
import glob
import xml.etree.ElementTree as et
from xml.dom import minidom
import argparse
import random

def addShape(root, name ):
    shape = et.SubElement(root, 'shape' )
    shape.set('id', '{0}_object'.format(name) )
    shape.set('type', 'ply' )
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

def addEnv(root, envmapName, scaleFloat):
    emitter = et.SubElement(root, 'emitter')
    emitter.set('type', 'envmap')
    filename = et.SubElement(emitter, 'string')
    filename.set('name', 'filename')
    filename.set('value', envmapName )
    scale = et.SubElement(emitter, 'float')
    scale.set('name', 'scale')
    scale.set('value', '%.4f' % (scaleFloat) )
    return root



parser = argparse.ArgumentParser()
parser.add_argument('--imNum', type=int, default=10, help='the number of camera view')
parser.add_argument('--imWidth', type=int, default=480, help='image width')
parser.add_argument('--imHeight', type=int, default=360, help='image height')
parser.add_argument('--sampleCount', type=int, default=512, help='the number of samples')
parser.add_argument('--fov', type=float, default=63.4149, help='the field of view in x axis')
parser.add_argument('--dist', type=float, default=3.0, help='the distance from camera to the target point')
parser.add_argument('--mode', default='train')
parser.add_argument('--envRoot', required=True, help='absolute path to environmental map root')
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
mode = opt.mode
fileRoot = opt.fileRoot
envRoot = opt.envRoot
intIOR = opt.intIOR
extIOR = opt.extIOR

shapes = glob.glob(osp.join(fileRoot, mode, 'Shape*') )
envmapNames = glob.glob(osp.join(envRoot, mode, '*.hdr') )
print('Total Number of %d envmap' % len(envmapNames ) )
np.random.seed(rs )


for n in range(rs, min(re, len(shapes)) ):
    shapeRoot = osp.join(fileRoot, mode, 'Shape__%d' % n )
    print('%d/%d: %s' % (n, min(re, len(shapes) ), shapeRoot ) )

    if osp.isfile( osp.join(shapeRoot, 'cam.txt' ) ):
        print("Warning: %s already exisits." % shapeRoot )
        continue

    # Create rendering file for Depth maps
    root = et.Element('scene')
    root.set('version', '0.5.0')
    integrator = et.SubElement(root, 'integrator')
    integrator.set('type', 'path')

    # Add shape
    root = addShape(root, 'poissonSubd.ply' )

    # Add environment light
    envId = (n-rs) % len(envmapNames )
    if envId == 0:
        random.shuffle(envmapNames )
    envmapName = envmapNames[envId ]
    scaleFloat = np.random.normal(5, 2)
    scaleFloat = max(scaleFloat, 0.8)
    root = addEnv(root, envmapName, scaleFloat)

    # Add sensor
    root = addSensor(root, fovValue, imWidth, imHeight, sampleCount)

    # Output xml file
    xmlString = transformToXml(root )
    with open(osp.join(shapeRoot, 'im.xml'), 'w') as xmlOut:
        xmlOut.write(xmlString )
