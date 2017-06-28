#
# Copyright 2017 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Do NOT execute this file unless you have the SunnyBrook data in a subdir!

import dicom, lmdb, cv2, re, sys
import os, fnmatch, shutil, subprocess
from IPython.utils import io
import numpy as np
import tensorflow as tf
np.random.seed(1234)
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'Greys_r'
#matplotlib inline
import warnings
from PIL import Image
warnings.filterwarnings('ignore') # we ignore a RuntimeWarning produced from dividing by zero


tf.app.flags.DEFINE_string('directory', '/Users/seongjungkim/PycharmProjects/LV_segmentation/exercise_solutions/data',
                           'Directory to write the tf files ' )

FLAGS = tf.app.flags.FLAGS

print("\nSuccessfully imported packages, hooray!\n")


SAX_SERIES = {
    # challenge training
    "SC-HF-I-1": "0004",
    "SC-HF-I-2": "0106",
    "SC-HF-I-4": "0116",
    "SC-HF-I-40": "0134",
    "SC-HF-NI-3": "0379",
    "SC-HF-NI-4": "0501",
    "SC-HF-NI-34": "0446",
    "SC-HF-NI-36": "0474",
    "SC-HYP-1": "0550",
    "SC-HYP-3": "0650",
    "SC-HYP-38": "0734",
    "SC-HYP-40": "0755",
    "SC-N-2": "0898",
    "SC-N-3": "0915",
    "SC-N-40": "0944",
}

SUNNYBROOK_ROOT_PATH = "./sunnybrook2009/"

TRAIN_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            "Sunnybrook Cardiac MR Database ContoursPart3",
                            "TrainingDataContours")
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        "challenge_training")

def shrink_case(case):
    toks = case.split("-")
    def shrink_if_number(x):
        try:
            cvt = int(x)
            return str(cvt)
        except ValueError:
            return x
    return "-".join([shrink_if_number(t) for t in toks])

class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r"/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-icontour-manual.txt", ctr_path)
        self.case = shrink_case(match.group(1))
        self.img_no = int(match.group(2))
    
    def __str__(self):
        return "<Contour for case %s, image %d>" % (self.case, self.img_no)
    
    __repr__ = __str__

def load_contour(contour, img_path):
    filename = "IM-%s-%04d.dcm" % (SAX_SERIES[contour.case], contour.img_no)
    full_path = os.path.join(img_path, contour.case, filename)
    f = dicom.read_file(full_path)
    img = f.pixel_array.astype(np.int)
    ctrs = np.loadtxt(contour.ctr_path, delimiter=" ").astype(np.int)
    if __debug__==True:
        print '#####def load_contour####'
        print 'filename',filename
        print 'length contours',len(ctrs)
        print '#########################'
        #print (ctrs);
    label = np.zeros_like(img, dtype="uint8")
    cv2.fillPoly(label, [ctrs], 1)
    return img, label
    
def get_all_contours(contour_path):
# walk the directory structure for all the contour files
    contours = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files, 'IM-0001-*-icontour-manual.txt')]
    print("Shuffle data")
# shuffle the data
    np.random.shuffle(contours)
    print("Number of examples: {:d}".format(len(contours)))
# apply the Contour class to each contour in the list giving each one a path
# case and img_no
    extracted = map(Contour, contours)
    return extracted

# helper functions included with at TF MNIST example online

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def export_all_contours(contours, img_path, img_name):
    counter_img = 0
    counter_label = 0
    batchsz = 100
    print("Processing {:d} images and labels...".format(len(contours)))
    filename = os.path.join(FLAGS.directory, img_name + '.tfrecords')
    print('Writing', filename)
    print 'int(np.ceil(len(contours) / float(batchsz)))',int(np.ceil(len(contours) / float(batchsz)))
    writer = tf.python_io.TFRecordWriter(filename)
    """
    for i in xrange(int(np.ceil(len(contours) / float(batchsz)))):
        batch = contours[(batchsz*i):(batchsz*(i+1))]
        if __debug__ ==True:
            print 'batch',batch
            print 'i',i
        if len(batch) == 0:
            break
    """
    imgs, labels = [], []

    for idx,ctr in enumerate(contours[:]):
        try:
            img, label = load_contour(ctr, img_path)
            imgs.append(img)
            labels.append(label)
            rows = img.shape[0]
            cols = img.shape[1]
            depth = 1
            img_raw = img.tostring()
            re_image = np.fromstring(img_raw, dtype=np.uint64)
            label_raw = label.tostring()
            re_label = np.fromstring(label_raw, dtype=np.uint8)

            if __debug__ ==True:
                print 'idx',idx
                print 'recover image shape',np.shape(re_image)
                print 'recover image shape',np.shape(re_label)
                print 're_image',np.shape(re_image.reshape(rows , cols))

                re_img=re_image.reshape(rows, cols)
                #plt.imshow(re_img)
                #plt.show()
                print 'image shape ',np.shape(img)
                print 'label shape' , np.shape(label)
                print 'rows',img.shape[0]
                print 'cols',img.shape[1]
                print 'depth',1

            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'height': _int64_feature(rows),
                    'width': _int64_feature(cols),
                    'depth': _int64_feature(depth),
                    'img_raw': _bytes_feature(img_raw),
                    'label_raw': _bytes_feature(label_raw)}))

            writer.write(example.SerializeToString())
#                print(type(img_raw))
#                print(len(imgs),len(labels))
#                if idx % 50 == 0:
#                    print ctr
#                    plt.imshow(img)
#                    plt.show()
#                    plt.imshow(label)
#                    plt.hist(img.ravel(),65536,[0,65536])
#                    plt.show()
        except IOError:
            continue
    writer.close()
def reconstruct_tfrecord_rawdata(tfrecord_path):

    print 'now Reconstruct Image Data please wait a second'
    reconstruct_image=[]
    #caution record_iter is generator

    record_iter = tf.python_io.tf_record_iterator(path = tfrecord_path)
    n=len(list(record_iter))
    record_iter = tf.python_io.tf_record_iterator(path = tfrecord_path)

    print 'The Number of Data :' , n
    ret_img_list=[]
    ret_lab_list=[]
    for i, str_record in enumerate(record_iter):
        msg = '\r -progress {0}/{1}'.format(i,n)
        sys.stdout.write(msg)
        sys.stdout.flush()

        example = tf.train.Example()
        example.ParseFromString(str_record)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        depth = int(example.features.feature['depth'].int64_list.value[0])
        raw_image = (example.features.feature['img_raw'].bytes_list.value[0])
        raw_label = (example.features.feature['label_raw'].bytes_list.value[0])

        image = np.fromstring(raw_image , dtype = np.uint64)
        label = np.fromstring(raw_label , dtype = np.uint8)
        image = image.reshape((height , width,-1))
        label = label.reshape((height , width , -1 ))
        ret_img_list.append(image)
        ret_lab_list.append(label)


        #ret_lab_list.append(label)
    ret_img=np.asarray(ret_img_list)
    ret_lab=np.asarray(ret_lab_list)
    return ret_img , ret_lab

def show_train_imgs_labs():
    imgs, labs = reconstruct_tfrecord_rawdata('./train_images.tfrecords')
    print np.shape(imgs)
    print np.shape(labs)
    for i in range(len(imgs)):
        img = imgs[i].squeeze()
        plt.imshow(img)
        plt.show()

        label = labs[i].squeeze()
        plt.imshow(label)
        plt.show()
def show_val_imgs_labs():
    imgs, labs = reconstruct_tfrecord_rawdata('./val_images.tfrecords')
    print np.shape(imgs)
    print np.shape(labs)
    for i in range(len(imgs)):
        img = imgs[i].squeeze()
        plt.imshow(img)
        plt.show()

        label = labs[i].squeeze()
        plt.imshow(label)
        plt.show()
def onehot2cls_image(image):
    tmp=np.zeros([256,256])
    for i in range(256):
        for j in range(256):
            ind=np.argmax(image[i,j,:] ,axis=0)
            tmp[i,j]=ind
    print np.shape(tmp)
    #tmp=tmp.astype(np.int32)
    print tmp
    tmp=(tmp*255)
    tmp=Image.fromarray(tmp)
    if tmp.mode != 'RGB':
        tmp = tmp.convert('RGB')
    tmp.save('./sample_pred.png')
    plt.imshow(tmp)

if __name__== "__main__":

    """
    SPLIT_RATIO = 0.1
    print("Mapping ground truth contours to images...")
    ctrs = get_all_contours(TRAIN_CONTOUR_PATH)
    val_ctrs = ctrs[0:int(SPLIT_RATIO*len(ctrs))]
    train_ctrs = ctrs[int(SPLIT_RATIO*len(ctrs)):]
    print("Done mapping ground truth contours to images")
    print("\nBuilding .tfrecord for train...")
    export_all_contours(train_ctrs, TRAIN_IMG_PATH, "train_images")

    print("\nBuilding .tfrecord for val...")
    export_all_contours(val_ctrs, TRAIN_IMG_PATH, "val_images")
    """


