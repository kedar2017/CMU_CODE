import numpy
import pylab
from PIL import Image

img = Image.open(open('/home/kedar/ccn/ml_task/labeled/0020.jpg'))
img= numpy.asarray(img,dtype='float64')/256

img_ =img.transpose(2,0,1).reshape(1,3,512,620)

#this image must be an input to the convolve and pooling layer
test_model = theano.function(
        [index],
        layer.errors(y),
        givens={
            x: test_set_x[index*batch_size:(index+1)*batch_size],
            y: test_set_y[index*batch_size:(index+1)*batch_size]
#In the training defined in the code the train_x is original pixels
#Vs train_y is labeled pixels

#U take one image(which is there both in the /left and /labeled
#Then put them into x and y respectively of train

#By this your weights will keep on varying

#Simultaneously put the data which is not learnt into the test and 
#validate
