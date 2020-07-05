import os.path as osp
import numpy as np
import glob
import xml.etree.ElementTree as et
from xml.dom import minidom
import argparse

def transformToXml(root ):
    rstring = et.tostring(root, 'utf-8')
    pstring = minidom.parseString(rstring)
    xmlString = pstring.toprettyxml(indent="    ")
    xmlString= xmlString.split('\n')
    xmlString = [x for x in xmlString if len(x.strip()) != 0 ]
    xmlString = '\n'.join(xmlString )
    return xmlString

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train')
parser.add_argument('--fileRoot', default='./Shapes/', help='path to the file root')
parser.add_argument('--rs', default = 0, type=int, help='the starting point')
parser.add_argument('--re', default = 10, type=int, help='the end point')
parser.add_argument('--camNum', type=int, required=True, help='number of cameras')
opt = parser.parse_args()
print(opt )

rs = opt.rs
re = opt.re
mode = opt.mode
fileRoot = opt.fileRoot

shapeFiles = glob.glob(osp.join(fileRoot, mode, 'Shape*') )

for n in range(rs, min(re, len(shapeFiles) ) ):
    shapeRoot = osp.join(fileRoot, mode, 'Shape__%d' % n )
    print('%d/%d: %s' % (n, min(re, len(shapeFiles) ), shapeRoot ) )

    xmlFile = osp.join(shapeRoot, 'im.xml')
    if not osp.isfile(xmlFile ):
        print('Warning: %s does not have xml file template.' % shapeRoot )
        #continue

    # Create rendering file for Depth maps
    tree = et.parse(xmlFile )
    root = tree.getroot()

    shapes = root.findall('shape' )
    assert(len(shapes ) == 1 )
    for shape in shapes:
        strings = shape.findall('string')
        assert(len(strings) == 1 )
        for st in strings:
            shapeFileName = st.get('value')
            assert(shapeFileName == 'poissonSubd.ply')
            st.set('value', 'visualHullSubd_%d.ply' % opt.camNum )


    # Output xml file
    xmlString = transformToXml(root )
    with open(osp.join(shapeRoot, 'imVH_%d.xml' % opt.camNum), 'w') as xmlOut:
        xmlOut.write(xmlString )

