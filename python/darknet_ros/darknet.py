from ctypes import *
import math
import random
import os
import tempfile
import re
import numpy as np

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr


class Darknet:

    def __init__(self, base_directory):
        self.base_directory = base_directory
        self.lib = CDLL(os.path.join(base_directory, "libdarknet.so"), RTLD_GLOBAL)
        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.predict = self.lib.network_predict
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        self.set_gpu = self.lib.cuda_set_device
        self.set_gpu.argtypes = [c_int]

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes = self.lib.make_network_boxes
        self.make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = self.lib.network_predict
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.do_nms_obj = self.lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

    def classify(self, net, meta, im):
        out = self.predict_image(net, im)
        res = []
        for i in range(meta.classes):
            res.append((meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    def detect(self, net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
        if isinstance(image, bytes):  
            # image is a filename 
            # i.e. image = b'/darknet/data/dog.jpg'
            im = self.load_image(image, 0, 0)
        else:  
            # image is an nparray
            # i.e. image = cv2.imread('/darknet/data/dog.jpg')
            im, image = array_to_image(image)
            self.rgbgr_image(im)

        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(net, im)
        dets = self.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if (nms):
            self.do_nms_obj(dets, num, meta.classes, nms)

        res = []
        for j in range(num):
            for i in range(meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))

        res = sorted(res, key=lambda x: -x[1])
        if isinstance(image, bytes):
            free_image(im)
        self.free_detections(dets, num)
        return res

    def get_temp_data(self, original_data_file, names_file='data/coco.names'):
        temp_data_file = tempfile.NamedTemporaryFile(delete=False)
        with open(os.path.join(self.base_directory, original_data_file), 'r') as f:
            data = f.read()
            replacement_data = re.sub(r'(\S*\.names)', os.path.join(self.base_directory, names_file), data)
            temp_data_file.write(replacement_data)
            temp_data_file.close()
        return temp_data_file

if __name__ == "__main__":
    base_directory = os.path.dirname(os.path.abspath(__file__)) + '/../..'
    darknet_instance = Darknet(base_directory)
    temp_data_file = darknet_instance.get_temp_data('cfg/coco.data')

    net = darknet_instance.load_net(os.path.join(base_directory, "cfg/yolov3.cfg"), os.path.join(base_directory, "yolov3.weights"), 0)
    meta = darknet_instance.load_meta(temp_data_file.name)
    os.remove(temp_data_file.name)
    r = darknet_instance.detect(net, meta, os.path.join(base_directory, "data/dog.jpg"))
    print r
