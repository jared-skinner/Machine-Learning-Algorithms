import gzip
import struct
import numpy as np

with gzip.open("/home/jared/mnist/train-labels-idx1-ubyte.gz", 'rb') as trainig_labels:
    line = trainig_labels.read(8)
    magic, num = struct.unpack(">II", line)

    print(magic, num)


#trainig_images = gzip.open("/home/jared/mnist/train-images-idx3-ubyte.gz", 'rb')


