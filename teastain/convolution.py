import math
import scipy.signal
import numpy as np

from PIL import Image


def MakeGaussian(mean, variance):
  ''' Returns a callable which takes a 2-tuple (an x,y coordinate point), and
  returns the value of a gaussian function with the given parameters at the
  given point.
  Args:
    mean:
        2-tuple; mu parameter to the gaussian function. Is the "center" of the
        mound.
    variance:
        2-tuple; sigma parameter to the gaussian function.
  '''
  # This is (2 * standard_deviation). Avoids computing constants in loops.
  double_stddev = [2.*float(v)**2 for v in variance]
  mean = [float(v) for v in mean]  # work in floats

  def Gaussian(point):
    ''' Returns the value of a 2d gaussian at the given point. '''
    return math.exp(-1. *
        sum((point[idx] - mean[idx])**2 / double_stddev[idx] for idx in [0,1]))

  return Gaussian


def MatrixFromFunction(size, scale, func):
  ''' Returns a 2d np matrix representing a discreteization of the given
  function. (0,0) on the coordinate axis is defined as the center of the matrix.
  Args:
    size:
        2-tuple; the size of the result matrix
    scale:
        2-tuple; the 'length' of the coordinate axis along each dimension
        projected onto the result matrix.
  '''
  scale = [float(v) for v in scale]  # work in floats
  def MatrixGenerator():
    # row/col corresponding to the (0,0) point in function space
    half_size = [(size[0] - 1) / 2., (size[1] - 1) / 2.]
    scale_factor = [scale[0] / float(size[0]), scale[1] / float(size[1])]
    for row in xrange(size[0]):
      for col in xrange(size[1]):
        # Transform a row,col into an x,y coordinate in function space
        point = [
            (row - half_size[0]) * scale_factor[0],
            (col - half_size[1]) * scale_factor[1],
        ]
        yield func(point)

  data = np.fromiter(MatrixGenerator(), float, size[0] * size[1])
  return np.reshape(data, size)


def ConvolveImg(img_in, convolve_mat):
  ''' Convolves the given image with a the given convolution matrix. The input
  matrix should be 3-dimensional with the depth dimension being equal to the
  number of channels in the image. This way, each dimension of `convolve_mat` is
  used to convolve a particular channel of the image.
  Args:
    img_in: PIL image to process
    convolve_mat: numpy array to convolve it with

  Returns:
    A new PIL.Image corresponding to the convolved image.
  '''
  def NormalizeTo(arr, new_max):
    ''' Normalizes a numpy array to the range [0, new_max]. Modifies the array
    in-place. Returns a reference to the (modified) input array.
    '''
    m = arr.max()
    if m != 0:
      arr /= m
      arr *= new_max
    return arr

  img_in_data = np.array(img_in)
  img_out_data = np.dstack([
      np.uint8(NormalizeTo(scipy.signal.convolve2d(img_in_data[...,idx],
                                                   convolve_mat[...,idx],
                                                   mode='same'),
                           255))
      for idx in xrange(img_in_data.shape[2])
  ])
  return Image.fromarray(img_out_data, img_in.mode)
