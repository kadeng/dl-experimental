
# coding: utf-8

# Install ffmpeg and libavcodec-dev (plus related stuff) from
# https://launchpad.net/~mc3man/+archive/ubuntu/trusty-media
# 
# Install OpenCV using the instructions from 
# http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/
# 
# In my case I had a build error related to missing CUDA libs, so the build command I used was
# 
#     cmake -D CMAKE_BUILD_TYPE=RELEASE \
#             -D CMAKE_INSTALL_PREFIX=/usr/local \
#             -D INSTALL_C_EXAMPLES=OFF \
#             -D INSTALL_PYTHON_EXAMPLES=ON \
#             -D "OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules" \
#             -D BUILD_EXAMPLES=ON \
#             -D WITH_CUDA=OFF ..
# 
# 

# 

# In[2]:



# In[6]:

from pytube import YouTube

# not necessary, just for demo purposes.
from pprint import pprint

#yt = YouTube("https://www.youtube.com/watch?v=yim23IFJOyQ")

#yt.set_filename('Chimpanzees1')
# Once set, you can see all the codec and quality options YouTube has made
# available for the perticular video by printing videos.

# pprint(yt.get_videos())


# In[7]:

#video = yt.get('mp4', '360p')


# In[8]:

# In this case, we'll need to specify both the codec (mp4) and resolution
# (either 360p or 720p).

# Okay, let's download it!
#video = yt.get('mp4', '360p')
#video.download('./videos/')


# In[9]:

# Now let's try to read this using our fresh opencv build..


# In[10]:

import cv2
import h5py


# In[11]:

from IPython.display import HTML, FileLink, display
from base64 import b64encode
def link_video(path):
    src = FileLink(path)
    video_tag = '<video controls alt="test" src="%s">' % (path)
    
    display(HTML(data=video_tag))


# In[12]:

vc = cv2.VideoCapture('videos/Chimpanzees1.mp4')
frames = 0
shape = None
dtype = None
while (True):
    ret, img = vc.read()
    # img should be an numpy ndarray of shape (width, heigth, channels)
    if (not ret):
        break
    if (shape is None):
        shape = img.shape
    if (dtype is None):
        dtype = img.dtype
    frames+=1
    
print "Could read %d frames of shape %r and dtype %r" % (frames, shape, dtype)


# In[13]:

#import h5py

#h5f = h5py.File('videos/Chimpanzees1_7.hdf5',driver='core', backing_store=True)
#h5f = h5py.File('/tmp/Chimpanzees1_7.hdf5')
#h5g = h5f.create_group('input')


# In[14]:

import numpy as np


# In[15]:

shape


# In[16]:

rshape = [ shape[2],shape[0], shape[1] ]


# In[17]:

ishape = tuple([ frames ] + list(rshape))
ichunkshape = tuple([ 1 ] + list(rshape))
imaxshape = tuple([ None ] + list(rshape))


# In[18]:

import numpy as np
from numpy.random import permutation


# In[19]:

#inframes = h5g.create_dataset('raw_hsl_frames', 
#                             shape=ishape, dtype=np.float32, 
#                             chunks=ichunkshape, maxshape=imaxshape,
#                             fletcher32=True, fillvalue=0)
#
inframes = np.zeros(shape=ishape, dtype=np.float32)


# In[20]:

permuted_indices = list(range(frames))


# In[21]:

vc = cv2.VideoCapture('videos/Chimpanzees1.mp4')
frames = 0
shape = None
dtype = None
while (True):
    ret, img = vc.read()
    if (not ret):
        break
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    
    hsvimgf = hsvimg.astype(np.float32) / 255.0
    hsvimgf = hsvimgf.swapaxes(2,0).swapaxes(1,2)
    # img should be an numpy ndarray of shape (width, heigth, channels)
    inframes[permuted_indices[frames],:,:,:] = hsvimgf
    
    #hsvimg2 = (inframes[permuted_indices[frames],:,:,:] * 255.0).astype(np.uint8)
    
    #print np.sum(hsvimg-hsvimg2)
    #bgrimg = cv2.cvtColor(hsvimg2, cv2.COLOR_HSV2BGR)
    #print np.sum(img-bgrimg)
    
    #cv2.imwrite("videos/recoded1_%05d.png" % (frames), bgrimg)
    
    frames+=1
    if (frames % 50==0):
        print "Frame #%d" % (frames)
print "Read %d frames" % (frames)


# In[22]:

#inframesmem = np.array(inframes)


# In[23]:

#!rm videos/Chimpanzees1_recoded.mp4
#!ffmpeg -i imageseries/Chimpanzees1_%05d.png -c:v libx264 -b:v 400k -vf format=yuv420p -maxrate:v 500k -bufsize 1000k  -r 10 -preset medium videos/Chimpanzees1_recoded.mp4


# In[24]:

#link_video('videos/Chimpanzees1_recoded.mp4')


# In[25]:

import os
import h5py
from glob import glob
import numpy as np


# ## MXNet ##
# 
# 

# In[26]:

import logging
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logging.debug("test")


# In[27]:

logger = logging.getLogger()


# In[ ]:




# In[28]:

import mxnet as mx
import mxnet.io as mxio


# In[29]:

logging.debug("test")


# In[30]:

a = mx.nd.ones((2,3), mx.cpu())


# In[31]:

b = mx.nd.zeros((2,3), mx.cpu())


# In[32]:

a[1:] = 2
a.copyto(b)


# In[33]:

b.asnumpy()


# In[34]:

net = mx.symbol.Variable('data')
y = mx.symbol.Variable('y')
dil = 1

for d in range(1,2):
    net = mx.symbol.Convolution(net, kernel=(3,3), num_filter=200, dilate=(1,1), name='conv%d' % (d))
    net = mx.symbol.LeakyReLU(net, act_type='prelu')
    net = mx.symbol.DropoutAllchannels(net, p=0.5)
    dil *= 2

net = mx.symbol.Convolution(net, kernel=(3,3), num_filter=3, name='decoder')
out = mx.symbol.MAERegressionOutput(data=net, label=y, name='output_')

def get_output_resolution(input_shape=(10, 3, 640, 480)):
    return net.infer_shape(data=input_shape)[-2][0][2:]

def get_middle_slice(X):
    res = get_output_resolution(X.shape)
    dx = (X.shape[2]-res[0]) // 2
    dy = (X.shape[3]-res[1]) // 2
    return X[:,:,dx:X.shape[2]-dx,dy:X.shape[3]-dy]



# In[35]:

num_epoch = 1
model = mx.model.FeedForward(ctx=mx.cpu(), symbol=out, num_epoch=num_epoch,
                             learning_rate=0.05, momentum=0.9, wd=0.00001)


# In[36]:

inframes.shape


# In[37]:

train_X = inframes[:400]
train_y = get_middle_slice(train_X)
test_X = inframes[1420:]
test_y = get_middle_slice(test_X)


# In[38]:

batch_size = 4
train_iter = mx.io.NDArrayIter({'data' : train_X}, { 'y' : train_y }, shuffle=True, batch_size=batch_size)
test_iter = mx.io.NDArrayIter({'data' : test_X}, { 'y' : test_y}, shuffle=True, batch_size=batch_size)


# In[39]:

train_iter.reset()
for batch in train_iter:
    print batch.data
    print batch.label
    break


# In[ ]:

model.fit(train_iter,
          eval_data=test_iter,
          eval_metric="mae",
          batch_end_callback=mx.callback.Speedometer(batch_size))


# In[58]:

get_ipython().magic(u'debug')


# In[ ]:

mx.viz.plot_network(symbol=net)


# In[ ]:

out.list_arguments()


# In[ ]:

arg_shape, out_shape, aux_shape = out.infer_shape(X=(10, 3, 640,480), y=(10,3,270,110))
dict(zip(net.list_arguments(), arg_shape))
size = 0
for s in arg_shape:
    print s
    size += np.prod(s)
print "Total parameter size=%d MB, total temp size=%d MB" % (size / (1024*1024), 640*480*1000*8 / (1024*1024))


# In[ ]:

net.list_arguments()


# Important Paper about Dilated Convolution:
# 
# http://arxiv.org/pdf/1511.07122v1.pdf
# 

# 
# ## Keras Models ##
# 

# In[ ]:


import numpy as np
np.random.seed(1337)  # for reproducibility


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.advanced_activations import ThresholdedReLU
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.datasets import imdb
from keras.optimizers import SGD
from keras.regularizers import l1,l2,l1l2, ActivityRegularizer, activity_l1
from keras import backend as K
import keras.layers.convolutional as kconv
import tensorflow as tf
import tensorflow as tf
from keras.layers import MaskedLayer
import keras.backend as K
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization

_FLOATX = K.floatx()


# In[ ]:

print 'Build model...X'

num_feature_detectors=12*12*3
tile_size=12
receptive_field_size=16
model = Sequential()
model.add(Dropout(0.001, input_shape=ichunkshape[1:]))

input_features = Convolution2D( name='encode_tiled_features', 
                        nb_filter=num_feature_detectors, nb_row=receptive_field_size,nb_col=receptive_field_size, 
                        subsample=(tile_size,tile_size), 
                        border_mode='valid', trainable=True,
                        dim_ordering='tf' )


print np.prod(ichunkshape[1:])
model.add(input_features)

print input_features.output_shape 

model.add(Activation('relu'))

decoder = Convolution2D( name='decode_tiled_features', 
                        nb_filter=tile_size*tile_size*3, nb_row=1,nb_col=1, subsample=(1,1),
                        border_mode='same', trainable=True,
                        dim_ordering='tf')

model.add(decoder)
model.add(Reshape(tuple(list(decoder.output_shape[1:3]) + [tile_size, tile_size, 3])))
print model.output_shape
model.add(Permute((1,3,2,4,5)))

print model.output_shape
model.add(Reshape((model.output_shape[1]*model.output_shape[2],model.output_shape[3]*model.output_shape[4],model.output_shape[5])))

model.add(ZeroPadding2D(((ichunkshape[1]-model.output_shape[1])/2, (ichunkshape[2]-model.output_shape[2])/2), 
                        dim_ordering='tf'))

print model.output_shape
def mean_abs_error_all(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=None)



print model.output_shape


# In[ ]:

model.compile(optimizer='adam', loss=mean_abs_error_all, class_mode='binary')


# In[ ]:

def data_generator(batchsize):
    for i in range(10):
        for j in permutation(1300):
            bi = inframesmem[j:(j+batchsize),:,:,:]

            yield bi,  bi

for i in range(1):
    print "Over-Epoch %i" % (i)
    model.fit_generator(data_generator(40), validation_data=(inframes[1400:1430], inframes[1400:1430]),
              show_accuracy=False, nb_epoch=1, samples_per_epoch=500, nb_worker=1)

model.save_weights('model1.hdf5')


# In[ ]:

def data_generator(batchsize):
    for i in range(10):
        for j in permutation(1300):
            bi = inframesm[j:(j+batchsize),:,:,:]
            yield bi,  bi

for i in range(1):
    print "Over-Epoch %i" % (i)
    model.fit_generator(data_generator(4), validation_data=(inframes[1400:1430], inframes[1400:1430]),
              show_accuracy=False, nb_epoch=5, samples_per_epoch=400, nb_worker=1)


# In[ ]:

print 'Build model...X'

num_feature_detectors=16*16*4
tile_size=16
receptive_field_size=24
model = Sequential()
model.add(Dropout(0.1, input_shape=ichunkshape[1:]))

input_features = Convolution2D( name='encode_tiled_features', 
                        nb_filter=num_feature_detectors, nb_row=receptive_field_size,nb_col=receptive_field_size, 
                        subsample=(tile_size,tile_size), 
                        border_mode='valid', trainable=True,
                        dim_ordering='tf', activity_regularizer=activity_l1(0.0000005))


model.add(input_features)
model.add(PReLU())

decoder = Convolution2D( name='decode_tiled_features', 
                        nb_filter=tile_size*tile_size*3, nb_row=1,nb_col=1, subsample=(1,1),
                        border_mode='same', trainable=True,
                        dim_ordering='tf')

model.add(decoder)
model.add(Reshape(tuple(list(decoder.output_shape[1:3]) + [tile_size, tile_size, 3])))
print model.output_shape
model.add(Permute((1,3,2,4,5)))

print model.output_shape
model.add(Reshape((model.output_shape[1]*model.output_shape[2],model.output_shape[3]*model.output_shape[4],model.output_shape[5])))

model.add(ZeroPadding2D(((ichunkshape[1]-model.output_shape[1])/2, (ichunkshape[2]-model.output_shape[2])/2), 
                        dim_ordering='tf'))

print model.output_shape
def mean_abs_error_all(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=None)



print model.output_shape


# In[ ]:

model.compile(optimizer='adam', loss=mean_abs_error_all, class_mode='binary')


# In[ ]:

def data_generator(batchsize):
    for i in range(10):
        perm = permutation(1300)
        for i in range(len(perm)//batchsize):
            bi = inframesmem[perm[i*batchsize:(i+1)*batchsize],:,:,:]
            yield bi,  bi

for i in range(1):
    print "Over-Epoch %i" % (i)
    model.fit_generator(data_generator(30), validation_data=(inframes[1400:1430], inframes[1400:1430]),
              show_accuracy=False, nb_epoch=5, samples_per_epoch=400, nb_worker=1)


# In[ ]:




# In[ ]:

print 'Build model...X'

num_feature_detectors=12*12*3
tile_size=12
receptive_field_size=16
model = Sequential()
model.add(Dropout(0.1, input_shape=ichunkshape[1:]))

enc1 = Convolution2D( name='enc1', 
                        nb_filter=num_feature_detectors, nb_row=receptive_field_size,nb_col=receptive_field_size, 
                        subsample=(tile_size,tile_size), 
                        border_mode='valid', trainable=True,
                        dim_ordering='tf', activity_regularizer=activity_l1(0.0000005))


model.add(enc1)
model.add(PReLU())
model.add(Dropout(0.1))
# ---------------
tile_size2=2
num_feature_detectors2=tile_size2*tile_size2*num_feature_detectors 
receptive_field_size2=2

enc2 = Convolution2D( name='enc2', 
                        nb_filter=num_feature_detectors2, nb_row=receptive_field_size2,nb_col=receptive_field_size2, 
                        subsample=(tile_size2,tile_size2), 
                        border_mode='valid', trainable=True,
                        dim_ordering='tf', activity_regularizer=activity_l1(0.0))


model.add(enc2)
model.add(PReLU())

# --------------------

dec2 = Convolution2D( name='dec2', 
                        nb_filter=tile_size2*tile_size2*num_feature_detectors, nb_row=1,nb_col=1, subsample=(1,1),
                        border_mode='valid', trainable=True,
                        dim_ordering='tf')

model.add(dec2)
# We need to reshape and reorder dimensions, so we can, at least in theory, do symmetric, layer-wise training...
newshape2 = tuple(list(dec2.output_shape[1:3])+ [tile_size2, tile_size2, num_feature_detectors])
model.add(Reshape(newshape2))
model.add(Permute((1,3,2,4,5)))
blockshape2 = (model.output_shape[1]*model.output_shape[2], model.output_shape[3]*model.output_shape[4], model.output_shape[5])

model.add(Reshape(blockshape2))

# --------------------

dec1 = Convolution2D( name='dec1', 
                        nb_filter=tile_size*tile_size*3, nb_row=1,nb_col=1, subsample=(1,1),
                        border_mode='same', trainable=True,
                        dim_ordering='tf')
model.add(dec1)
newshape1 = tuple(list(dec1.output_shape[1:3])+ [tile_size, tile_size, 3])
model.add(Reshape(newshape1))
model.add(Permute((1,3,2,4,5)))

blockshape1 = (model.output_shape[1]*model.output_shape[2],model.output_shape[3]*model.output_shape[4], model.output_shape[5])

model.add(Reshape(blockshape1))

model.add(ZeroPadding2D(((ichunkshape[1]-model.output_shape[1])/2, (ichunkshape[2]-model.output_shape[2])/2), 
                        dim_ordering='tf'))

# --------------------

print model.output_shape
def mean_abs_error_all(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=None)



print model.output_shape


# In[ ]:

model.compile(optimizer='adam', loss=mean_abs_error_all, class_mode='binary')


# In[ ]:

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')


# In[ ]:

def data_generator(batchsize):
    while True:
        perm = permutation(1400)
        for i in range(len(perm)//batchsize):
            bi = inframesmem[perm[i*batchsize:(i+1)*batchsize],:,:,:]
            yield bi,  bi

for i in range(1):
    print "Over-Epoch %i" % (i)
    model.fit_generator(data_generator(40), validation_data=(inframes[1400:1480], inframes[1400:1480]),
              show_accuracy=False, nb_epoch=10, samples_per_epoch=500, nb_worker=1)


# In[ ]:

model.save_weights('sparse_autoencoder_4layers.hdf5')


# 
# ## Visualizing the reconstructed video ##

# In[ ]:

import cv2


# In[ ]:

get_ipython().system(u'rm imageseries/*.jpg')


# In[ ]:

j = 0
c = 0
for i in permuted_indices:
    
    
    #print np.sum(hsvimg-hsvimg2)
    fframe = inframes[i,:,:,:]
    frame = (fframe * 255.0).astype(np.uint8)
    
    bgrimg = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    
    pframe = model.predict(fframe[np.newaxis,:,:,:], batch_size=1)
    np.clip(pframe, 0.0, 1.0, pframe)
    #print mean_squared_error_all
    piframe = (pframe[0,:,:,:]*255.0).astype(np.uint8)
    rgbimg = cv2.cvtColor(piframe, cv2.COLOR_HSV2BGR)
    
    j+=1
    c+=1
    cv2.imwrite("imageseries/reconstruction1_%05d.jpg" % (c), rgbimg)
    
    if (j==800):
        for z in range(200):
            pframe = model.predict(pframe, batch_size=1)
            np.clip(pframe, 0.0, 1.0, pframe)
            #print mean_squared_error_all
            piframe = (pframe[0,:,:,:]*255.0).astype(np.uint8)
            rgbimg = cv2.cvtColor(piframe, cv2.COLOR_HSV2BGR)
            c+=1
            cv2.imwrite("imageseries/reconstruction1_%05d.jpg" % (c), rgbimg)
    if (j % 100==0):
        me = np.mean(np.square((bgrimg.astype(np.float32)/255.0)-(rgbimg.astype(np.float32)/255.0)))
        print "Wrote image %d - MSE: %f" % (j, me)


# In[ ]:

get_ipython().system(u'rm imageseries/*.png')


# In[ ]:

get_ipython().system(u'rm videos/Chimpanzees1_reconstructed2.mp4')
get_ipython().system(u'ffmpeg -i imageseries/reconstruction1_%05d.jpg -c:v libx264 -b:v 400k -maxrate:v 500k -bufsize 1000k -r 10 -preset medium videos/Chimpanzees1_reconstructed2.mp4')


# In[ ]:

get_ipython().system(u'ls -hs videos/*.mp4')


# In[ ]:

link_video('videos/Chimpanzees1_reconstructed2.mp4')
#link_video('videos/Chimpanzees1_reconstructed2.mp4')


# In[ ]:

display(FileLink('videos/Chimpanzees1_reconstructed2.mp4'))


# In[ ]:

class array(object) :
    """Simple Array object that support autodiff."""
    def __init__(self, value, name=None):
        self.value = value
        if name:
            self.grad = lambda g : {name : g}

    def __add__(self, other):
        assert isinstance(other, int)
        ret = array(self.value + other)
        ret.grad = lambda g : self.grad(g)
        return ret

    def __mul__(self, other):
        assert isinstance(other, array)
        ret = array(self.value * other.value)
        def grad(g):
            x = self.grad(g * other.value)
            x.update(other.grad(g * self.value))
            return x
        ret.grad = grad
        return ret

# some examples
a = array(1, 'a')
b = array(2, 'b')
c = b * a
d = c + 1
print d.value
print d.grad(1)


# In[ ]:

(b*a).grad(2)


# In[ ]:



