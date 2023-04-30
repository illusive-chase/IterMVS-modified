import numpy as np
import re
import sys
from PIL import Image
import cv2


def read_pfm(filename, flip=True):
    # rb: binary file and read only
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':  # depth is Pf
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8')) # re is used for matching
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width, 1)
    # depth: H*W
    data = np.reshape(data, shape)
    if flip:
        data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)
    # print(image.shape)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()

def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))

    depth_min = float(lines[11].split()[0])
    depth_max = float(lines[11].split()[-1])

    return intrinsics, extrinsics, depth_min, depth_max

def write_cam_file(filename, intrinsics, extrinsics, depth_min, depth_max):
    with open(filename, 'w') as f:
        f.write('extrinsic\n')
        for i in range(4):
            line = ' '.join([str(extrinsics[i, j]) for j in range(4)])
            f.write(line + '\n')
        f.write('\nintrinsic\n')
        for i in range(3):
            line = ' '.join([str(intrinsics[i, j]) for j in range(3)])
            f.write(line + '\n')
        f.write('\n{} {}\n'.format(depth_min, depth_max))

def read_img(filename, h, w):
    img = Image.open(filename)
    # scale 0~255 to -1~1
    np_img = 2*np.asarray(img, dtype=np.float32) / 255. - 1
    original_h, original_w, _ = np_img.shape
    np_img = cv2.resize(np_img, (w, h), interpolation=cv2.INTER_LINEAR)
    
    np_img_ms = {
        "level_3": cv2.resize(np_img, (w//8, h//8), interpolation=cv2.INTER_LINEAR),
        "level_2": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_LINEAR),
        "level_1": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_LINEAR),
        "level_0": np_img
    }
    return np_img_ms, original_h, original_w

def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) != 0:
                data.append((ref_view, src_views))
    return data