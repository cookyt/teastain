#! /usr/bin/env python2

import argparse
import numpy as np
import random

from PIL import Image
from teastain.convolution import MakeGaussian, MatrixFromFunction, ConvolveImg

DESCRIPTION = '''
A quick and dirty tool for testing teastain's ability to apply convolution
matrices to images.
'''

def ParseArguments():
  parser = argparse.ArgumentParser(description=DESCRIPTION)
  parser.add_argument('--input_image', '-i',
      required=True,
  )
  parser.add_argument('--output_image', '-o',
      required=True,
  )
  return parser.parse_args()


def Triplify(arr):
  ''' Takes an NxM matrix and turns it into a NxMx3 matrix by stacking 3 copies
  of `arr` on top of each other.
  '''
  return np.dstack([arr]*3)


def NormalizeSum(arr):
  ''' Normalizes the sum of all elements in the numpy array so that the sum is
  one. Works in-place. Returns a reference to the (modified) input array. Does
  nothing if the sum is zero.
  '''
  s = arr.sum()
  if s != 0:
    arr /= s
  return arr


def SplitChannels(img):
  ''' Splits image into several corresponding to the individual channels of the
  image. Returns a list of images.
  '''
  img_data = np.array(img)
  channels = [
    Image.fromarray(np.dstack([img_data[...,idx]]*(img_data.shape[2])),
                    img.mode)
    for idx in range(img_data.shape[2])
  ]
  return channels


def main():
  args = ParseArguments()

  mat_size = (16, 16)
  mat_scale = (16, 16)  # scale of projection from real-space to matrix-space
  gauss_cmat = Triplify(MatrixFromFunction(
      mat_size,
      mat_scale,
      MakeGaussian(
          (0, 0),  # mean
          (3, 3),  # variance
      ),
  ))
  add_cmat = np.dstack(map(NormalizeSum, [
      MatrixFromFunction(mat_size, mat_scale, lambda (x,y): x + y),
      MatrixFromFunction(mat_size, mat_scale, lambda (x,y): x - y),
      MatrixFromFunction(mat_size, mat_scale, lambda (x,y): y - x),
  ]))
  rand_cmat = np.dstack([MatrixFromFunction(
      mat_size,
      mat_scale,
      lambda _: random.random(),
  ) for __ in xrange(3)])

  # Wierd results for this one. ConvolveImg implementation might be flawed.
  sobel_x_mat = np.array([
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0],
  ])
  twisted_sobel_cmat = np.dstack([sobel_x_mat, sobel_x_mat.T, sobel_x_mat])
  sobel_x_cmat = Triplify(sobel_x_mat)
  sobel_y_cmat = Triplify(sobel_x_mat.T)

  # convolve_cmat = 0.1 * add_cmat + gauss_cmat
  convolve_cmat = sobel_x_cmat

  img_in = Image.open(args.input_image)
  img_in.draft(mode='RGB', size=img_in.size)

  img_out = ConvolveImg(img_in, convolve_cmat)
  img_out.save(args.output_image)

  # Save the result image's channels seperately.
  # img_out_channels = SplitChannels(img_out)
  # for idx, fname in enumerate(['red', 'green', 'blue']):
  #   img_out_channels[idx].save('out_%s.png' % fname)


if __name__ == '__main__':
  main()
