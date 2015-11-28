import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from PIL import Image
import glob
import cv2
import statistics
from matplotlib import pyplot as plt
train_ori_pixels = [[],[],[]]
train_lab_pixels = []
test_ori_pixels=[[],[],[]]
test_lab_pixels =[]
valid_ori_pixels=[[],[],[]]
valid_lab_pixels=[]
count =0
IMAGES=10
#####################################################Whitening##########################################
def whiten(X,fudge=1E-18):

   # the matrix X should be observations-by-components

   # get the covariance matrix
   Xcov = np.dot(X.T,X)

   # eigenvalue decomposition of the covariance matrix
   d, V = np.linalg.eigh(Xcov)

   # a fudge factor can be used so that eigenvectors associated with
   # small eigenvalues do not get overamplified.
   D = np.diag(1. / np.sqrt(d+fudge))

   # whitening matrix
   W = np.dot(np.dot(V, D), V.T)

   # multiply by the whitening matrix
   X_white = np.dot(X, W)

   return X_white, W
################################################################get_data()##############################
def get_data():
	global train_ori_pixels
	global train_lab_pixels
	global test_ori_pixels
	global test_lab_pixels
	global valid_ori_pixels
	global valid_lab_pixels
	global count
	global IMAGES
	iterator_x=[5*i for i in range(100)]
	iterator_y=[5*i for i in range(125)]
	count=0
	#########################Training data##############################Ori
	for filename in glob.glob('/home/kedar/CMU/ccn/ml_task/left/Testing/*.jpg'):
		img=Image.open(filename)
		img=numpy.asarray(img,dtype='float64')/256
		img=img.transpose(2,0,1)
		total=[]
		total_image=[]
		verti=[]
		for a in range(3):
			for i in iterator_x:
				for j in iterator_y:
					for k in range(8):
						for l in range(8):
							if(not(i+k>510) and not(j+l>635)):
								verti.append(img[a][i+k][j+l])
							else:
								verti=[]
					if(not len(verti)==0):
						#verti=whiten(verti)[0]
						total_image.append(numpy.asarray(verti))
					verti=[]
			#total_image=whiten(verti)[0]
			for i in numpy.asarray(total_image):
				train_ori_pixels[a].append(i)
			total_image=[]
		count=count+1
		if(count>IMAGES):
			break
	count=0
	train_ori_pixels=numpy.asarray(train_ori_pixels,dtype='float64')
	train_ori_pixels=train_ori_pixels.transpose(1,2,0)
	###########################Test and validate#############################Ori
	for filename in glob.glob('/home/kedar/CMU/ccn/ml_task/left/Testing/*.jpg'):
		img=Image.open(filename)
		img=numpy.asarray(img,dtype='float64')/256
		img=img.transpose(2,0,1)
		total=[]
		total_image=[]
		verti=[]
		for a in range(3):
			for i in iterator_x:
				for j in iterator_y:
					for k in range(8):
						for l in range(8):
							if(not(i+k>510) and not(j+l>635)):
								verti.append(img[a][i+k][j+l])
							else:
								verti=[]
					if(not len(verti)==0):
						#verti=whiten(verti)[0]
						total_image.append(numpy.asarray(verti))
					verti=[]
			#total_image=whiten(total_image)[0]
			for i in numpy.asarray(total_image):
				test_ori_pixels[a].append(i)
				valid_ori_pixels[a].append(i)
			total_image=[]
		count=count+1
		if(count>IMAGES):
			break
	
	test_ori_pixels=numpy.asarray(test_ori_pixels,dtype='float64')
	test_ori_pixels=test_ori_pixels.transpose(1,2,0)
	valid_ori_pixels=numpy.asarray(valid_ori_pixels,dtype='float64')
	valid_ori_pixels=valid_ori_pixels.transpose(1,2,0)
	count=0
	#############################Training #################################Labels
	print train_ori_pixels[1][0]
	for filename in glob.glob('/home/kedar/CMU/ccn/ml_task/labeled/Testing/*.png'):
		img=Image.open(filename)
		img=numpy.asarray(img,dtype='float64')
		verti=[]
		total=[]
		total_image=[]
		for i in iterator_x:
			for j in iterator_y:
				verti=[]
				for k in range(8):
					for l in range(8):
						if(not(i+k>510) and not(j+l>635)):
							verti.append(img[i+k][j+l])
						else:
							verti=[]
				#verti=whiten(verti)[0]
				if(not len(verti)==0):
					if(int(statistics.mean(verti))==0.0):
						total_image.append(0.0)
					else:
						if(int(statistics.mean(verti))>9.0):
							total_image.append(9.0)
						else:
							total_image.append(int(statistics.mean(verti)))
		count=count+1
		#total_image=whiten(total_image)[0]
		for i in total_image:
			train_lab_pixels.append(i)
		if(count>IMAGES):
			break
	count=0
	###############################Testing and validate##########################Labels
	for filename in glob.glob('/home/kedar/CMU/ccn/ml_task/labeled/Testing/*.png'):
		img=Image.open(filename)
		img=numpy.asarray(img,dtype='float64')
		verti=[]
		total=[]
		total_image=[]
		for i in iterator_x:
			for j in iterator_y:
				verti=[]
				for k in range(8):
					for l in range(8):
						if(not(i+k>510) and not(j+l>635)):
							verti.append(img[i+k][j+l])
						else:
							verti=[]
				#verti=whiten(verti)[0]
				if(not len(verti)==0):
					if(int(statistics.mean(verti))==0.0):
						total_image.append(1.0)
					else:
						if(int(statistics.mean(verti))>9.0):
							total_image.append(9.0)
						else:
							total_image.append(int(statistics.mean(verti)))
		count=count+1
		#total_image=whiten(total_image)[0]
		if(True):
			for i in total_image:
				test_lab_pixels.append(i)
			for i in total_image:
				valid_lab_pixels.append(i)
		if(count>IMAGES):
			break
	count=0
	print len(train_lab_pixels)
	print len(train_ori_pixels)

	'''
	test_pixels=[(ori_pixels,lab_pixels)]
	valid_pixels=[(ori_pixels,lab_pixels)]
	'''
########################################################Hidden Layer#####################################################
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

##############################################################Logistic Regression##########################################
class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1
        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return [T.mean(T.neq(self.y_pred, y)),self.y_pred]
        else:
            raise NotImplementedError()
	def pred(self,y):
		return self.y_pred


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

###################################################Convolution####################################################
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


def evaluate_lenet5(learning_rate=0.1, n_epochs=200,dataset='mnist.pkl.gz',nkerns=[20, 50], batch_size=500):
	""" Demonstrates lenet on MNIST dataset

	:type learning_rate: float
	:param learning_rate: learning rate used (factor for the stochastic
						  gradient)

	:type n_epochs: int
	:param n_epochs: maximal number of epochs to run the optimizer

	:type dataset: string
	:param dataset: path to the dataset used for training /testing (MNIST here)

	:type nkerns: list of ints
	:param nkerns: number of kernels on each layer
	"""
	global train_ori_pixels
	global train_lab_pixels
	global test_ori_pixels
	global test_lab_pixels
	global valid_ori_pixels
	global valid_lab_pixels
	global count
	rng = numpy.random.RandomState(23455)
	train_set_x=theano.shared(numpy.asarray(train_ori_pixels,dtype=theano.config.floatX),borrow=True)
	train_set_y=theano.shared(numpy.asarray(train_lab_pixels,dtype=theano.config.floatX),borrow=True)
	train_set_y=T.cast(train_set_y,'int32')
	test_set_x=theano.shared(numpy.asarray(test_ori_pixels,dtype=theano.config.floatX),borrow=True)
	test_set_y=theano.shared(numpy.asarray(test_lab_pixels,dtype=theano.config.floatX),borrow=True)
	test_set_y=T.cast(test_set_y,'int32')
	valid_set_x=theano.shared(numpy.asarray(valid_ori_pixels,dtype=theano.config.floatX),borrow=True)
	valid_set_y=theano.shared(numpy.asarray(valid_lab_pixels,dtype=theano.config.floatX),borrow=True)
	valid_set_y=T.cast(valid_set_y,'int32')
	############
	n_train_batches = train_set_x.get_value(borrow=True).shape[0]
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
	n_test_batches = test_set_x.get_value(borrow=True).shape[0]
	n_train_batches /= batch_size
	n_valid_batches /= batch_size
	n_test_batches /= batch_size
	print "n_train_batches", n_train_batches
    ###########
	# allocate symbolic variables for the data
	index = T.lscalar()  # index to a [mini]batch

	# start-snippet-1
	x = T.tensor3('x')   # the data is presented as rasterized images
	y = T.ivector('y')  # the labels are presented as 1D vector of
						# [int] labels
	
	######################
	# BUILD ACTUAL MODEL #
	######################
	print '... building the model'

	# Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
	# to a 4D tensor, compatible with our LeNetConvPoolLayer
	# (28, 28) is the size of MNIST images.
	layer0_input = x.reshape((batch_size, 3, 8, 8))

	# Construct the first convolutional pooling layer:
	# filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
	# maxpooling reduces this further to (24/2, 24/2) = (12, 12)
	# 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
	layer0 = LeNetConvPoolLayer(
		rng,
		input=layer0_input,
		image_shape=(batch_size, 3, 8, 8),
		filter_shape=(nkerns[0], 3, 3, 3),
		poolsize=(2, 2)
	)

	# Construct the second convolutional pooling layer
	# filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
	# maxpooling reduces this further to (8/2, 8/2) = (4, 4)
	# 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
	layer1 = LeNetConvPoolLayer(
		rng,
		input=layer0.output,
		image_shape=(batch_size, nkerns[0], 3, 3),
		filter_shape=(nkerns[1], nkerns[0], 2, 2),
		poolsize=(2, 2)
	)
	# the HiddenLayer being fully-connected, it operates on 2D matrices of
	# shape (batch_size, num_pixels) (i.e matrix of rasterized images).
	# This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
	# or (500, 50 * 4 * 4) = (500, 800) with the default values.
	layer2_input = layer1.output.flatten(2)

	# construct a fully-connected sigmoidal layer
	layer2 = HiddenLayer(
		rng,
		input=layer2_input,
		n_in=nkerns[1] * 1 * 1,
		n_out=500,
		activation=T.tanh
	)

	# classify the values of the fully-connected sigmoidal layer
	layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

	# the cost we minimize during training is the NLL of the model
	cost = layer3.negative_log_likelihood(y)

	# create a function to compute the mistakes that are made by the model
	test_model = theano.function(
		[index],
		layer3.errors(y),
		givens={
			x: test_set_x[index * batch_size: (index + 1) * batch_size],
			y: test_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)

	validate_model = theano.function(
		[index],
		layer3.errors(y)[0],
		givens={
			x: valid_set_x[index * batch_size: (index + 1) * batch_size],
			y: valid_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)
	pred_model = theano.function(
		[index],
		layer3.errors(y)[1],
		givens={
			x: valid_set_x[index * batch_size: (index + 1) * batch_size],
			y: valid_set_y[index * batch_size: (index + 1) * batch_size]
		},
		on_unused_input='ignore'
	)
	# create a list of all model parameters to be fit by gradient descent
	params = layer3.params + layer2.params + layer0.params

	# create a list of gradients for all model parameters
	grads = T.grad(cost, params)
	#######################################
	# train_model is a function that updates the model parameters by
	# SGD Since this model has many parameters, it would be tedious to
	# manually create an update rule for each model parameter. We thus
	# create the updates list by automatically looping over all
	# (params[i], grads[i]) pairs.
	updates = [
		(param_i, param_i - learning_rate * grad_i)
		for param_i, grad_i in zip(params, grads)
	]

	train_model = theano.function(
		[index],
		cost,
		updates=updates,
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)
	# end-snippet-1

	###############
	# TRAIN MODEL #
	###############
	print '... training'
	# early-stopping parameters
	patience = 10000  # look as this many examples regardless
	patience_increase = 2  # wait this much longer when a new best is
						   # found
	improvement_threshold = 0.995  # a relative improvement of this much is
								   # considered significant
	validation_frequency = min(n_train_batches, patience / 2)
								  # go through this many
								  # minibatche before checking the network
								  # on the validation set; in this case we
								  # check every epoch

	best_validation_loss = numpy.inf
	best_iter = 0
	test_score = 0.
	start_time = timeit.default_timer()

	epoch = 0
	done_looping = False

	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):

			iter = (epoch - 1) * n_train_batches + minibatch_index

			if iter % 100 == 0:
				print 'training @ iter = ', iter
			cost_ij = train_model(minibatch_index)

			if (iter + 1) % validation_frequency == 0:

				# compute zero-one loss on validation set
				validation_losses = [validate_model(i) for i
								     in xrange(n_valid_batches)]
				this_validation_loss = numpy.mean(validation_losses)
				print('epoch %i, minibatch %i/%i, validation error %f %%' %
					  (epoch, minibatch_index + 1, n_train_batches,
					   this_validation_loss * 100.))
				pred = []
				print "Prediction", pred_model(5)
				for i in xrange(n_test_batches):
					z=pred_model(i)
					for j in z:
						pred.append(j)
				iterator_x=[5*i for i in range(100)]
				iterator_y=[5*i for i in range(125)]
				counter=55000
				print "Length of prediction", len(pred)
				print "Test_batches", n_train_batches
				#################################################
				filename='/home/kedar/CMU/ccn/ml_task/labeled/Training/1060_left.png'
				img=Image.open(filename)
				img=numpy.asarray(img,dtype='float64')
				for i in iterator_x:
					for j in iterator_y:
						for k in range(8):
							for l in range(8):
								if(not(i+k>505) and not(j+l>635)):
									if(counter<len(pred)):
										img[i+k][j+l]=10*pred[counter]
									else:
										print "OUT_1"
										img[i+k][j+l]=0
								else:
									print "OUT_2"
									img[i+k][j+l]=0
						counter=counter+1
				img=numpy.asarray(img,dtype='float64')
				plt.imshow(img, interpolation='nearest')
				plt.show()
				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:

					#improve patience if loss improvement is good enough
					if this_validation_loss < best_validation_loss *  \
					   improvement_threshold:
						patience = max(patience, iter * patience_increase)

					# save best validation score and iteration number
					best_validation_loss = this_validation_loss
					best_iter = iter

					# test it on the test set
					test_losses = [test_model(i)[0] for i in xrange(n_test_batches)]
					test_score = statistics.mean(test_losses)					
					#################################################

					print(('     epoch %i, minibatch %i/%i, test error of '
						   'best model %f %%') %
						  (epoch, minibatch_index + 1, n_train_batches,
						   test_score * 100.))

			if patience <= iter:
				done_looping = True
				break

	end_time = timeit.default_timer()
	print('Optimization complete.')
	print('Best validation score of %f %% obtained at iteration %i, '
		  'with test performance %f %%' %
		  (best_validation_loss * 100., best_iter + 1, test_score * 100.))
	print >> sys.stderr, ('The code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
	get_data()
	evaluate_lenet5()


def experiment(state, channel):
	evaluate_lenet5(state.learning_rate, dataset=state.dataset)
