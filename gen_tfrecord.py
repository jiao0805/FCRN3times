import tensorflow as tf
import os
from PIL import Image
import numpy as np

height = 318
width = 424
depth_height=318
depth_width=424

# trainwriter= tf.python_io.TFRecordWriter("train.tfrecords")
# rgbdir = './rgb_train/'
# depthdir = './depth_train/'
# eptlist=[]  #
# for img_name in os.listdir(rgbdir):
#     #print(img_name)
#     if os.path.splitext(img_name)[1] == '.jpg':
#         if os.path.isfile(os.path.join(depthdir, img_name).replace(".jpg",".png")):
#             eptlist.append(img_name)
# lens=len(eptlist)
# print(lens)
# indexar=np.arange(lens)
# print(indexar)
# randindex=np.random.permutation(indexar)
# counter=0
# for indexx in randindex:
#     counter = counter + 1
#     print(counter)
#     filename = eptlist[indexx]
#     imgraw = Image.open(rgbdir+filename).convert('RGB')
#     imgraw = imgraw.resize((width, height),Image.BILINEAR)
#     imgraw = imgraw.tobytes()
#     imglabel = Image.open(depthdir+filename.replace(".jpg",".png")).convert('F')
#     imglabel = imglabel.resize((depth_width, depth_height),Image.BILINEAR)
#     imglabel = imglabel.tobytes()
#     example = tf.train.Example(features=tf.train.Features(feature={
#         'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgraw])),
#         'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imglabel]))
#     }))
#     trainwriter.write(example.SerializeToString())
# print("traindata over")
# trainwriter.close()

testwriter= tf.python_io.TFRecordWriter("test.tfrecords")
rgbdir = './rgb_test/'
depthdir = './depth_test/'
eptlist=[]
for img_name in os.listdir(rgbdir):
    if os.path.splitext(img_name)[1] == '.jpg':
        if os.path.isfile(depthdir+img_name.replace(".jpg",".png")):
            eptlist.append(img_name)
lens=len(eptlist)
print(lens)
indexar=np.arange(lens)
randindex=np.random.permutation(indexar)
counter=0
for indexx in randindex:

    print(counter)
    counter = counter + 1
    filename=eptlist[indexx]
    imgraw=Image.open(rgbdir+filename).convert('RGB')
    imgraw = imgraw.resize((width, height),Image.BILINEAR)
    imgraw = imgraw.tobytes()
    imglabel=Image.open(depthdir+filename.replace(".jpg",".png")).convert('F')
    imglabel = imglabel.resize((depth_width, depth_height),Image.BILINEAR)
    imglabel = imglabel.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgraw])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imglabel]))
    }))
    testwriter.write(example.SerializeToString())
print("testdata over")
testwriter.close()