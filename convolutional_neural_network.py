import numpy as np


class ConvNerualNet(TrainingModel):
    '''
    Basic implementation of a convolutional neural network
    '''

    def __init__(self):


        # generate filters for each convolutional layers


        pass




    def cost(self):
        pass






    def back_prop(self):
        pass


    def foward_feed(self):

        #loop through architecture



        # apply fully connected layer



        # output array of categories


        pass


    def pool(self):
        pass




    def apply_filter(self, X, filter, stride):
        '''
        apply a given filter to an input volume via a convolution like process
        '''

        # verify the filter has the same depth as X
        assert(X.shape[2] == filter.shape[2])

        # calculate the amount of padding X will need so that after applying the
        # convolution we don't mistakenly downsize our image
        col_padding = (filter.shape[0] - 1) / 2
        row_padding = (filter.shape[1] - 1) / 2

        padded_X = np.copy(X)

        # pad with zeros 
        row_pad = np.zeros([padded_X.shape[0], row_padding, padded_X.shape[2]])
        padded_X = np.concatenate(row_pad, padded_X, row_pad, axis=0)

        # generate this after padding rows, since this will change the shape of
        # X
        col_pad = np.zeros([col_padding, padded_X.shape[1], padded_X.shape[2]])
        padded_X = np.concatenate(col_pad, padded_X, col_pad, axis=1)

        # create result array
        result = np.zeros([X.shape[0]/stride, X.shape[1]/stride])

        # slide the filter across the image by stride
        # TODO: add in stride term
        for i in range(0, X.shape[0] - 1, stride):
            for j in range(0, X.shape[1] - 1, stride):
                # do the calculation and add value to resulting tensor
                result[i, j] = np.sum(padded_X[i:i + filter.shape[0], j:j + filter.shape[1]] * filter)

        return result


def main():
    pass







if __name__ == "__main__":



    # x will be a volume with dimensions width X height X channels

    main()

