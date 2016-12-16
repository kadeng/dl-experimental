from collections import OrderedDict
import numpy as np
from mxnet.io import DataIter, NDArrayIter, DataBatch, _init_data
from mxnet.ndarray import array
class NDArrayGeneratorIter(DataIter):
    
    """NDArrayIter object in mxnet. Taking NDArray or numpy array to get dataiter.
    Parameters
    ----------
    generator: Iterable providing tuples of (data,label) params as expected by NDArrayIter
    batch_size: int
        Batch Size
    shuffle: bool
        Whether to shuffle the data
    last_batch_handle: 'pad', 'discard' or 'roll_over'
        How to handle the last batch
    Note
    ----
    This iterator will pad, discard or roll over the last batch if
    the size of data does not match batch_size. Roll over is intended
    for training and can cause problems if used for prediction.
    """
    def __init__(self, generator, gen_count=10, shuffle=False, batch_size=1, last_batch_handle='pad'):
        super(NDArrayGeneratorIter, self).__init__()
        self.generator = generator
        self.gen_count = gen_count
        self.gen_idx = 0
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.last_batch_handle = last_batch_handle
        self._generate_next()
        self.regenerate = False
        
    def _reinit(self, data, label=None, batch_size=1, shuffle=False, last_batch_handle='pad'):    
        # pylint: disable=W0201
        self.data = _init_data(data, allow_empty=False, default_name='data')
        self.label = _init_data(label, allow_empty=True, default_name='softmax_label')

        # shuffle data
        if shuffle:
            idx = np.arange(self.data[0][1].shape[0])
            np.random.shuffle(idx)
            self.data = [(k, v[idx]) for k, v in self.data]
            self.label = [(k, v[idx]) for k, v in self.label]

        self.data_list = [x[1] for x in self.data] + [x[1] for x in self.label]
        self.num_source = len(self.data_list)

        # batching
        if last_batch_handle == 'discard':
            new_n = self.data_list[0].shape[0] - self.data_list[0].shape[0] % batch_size
            data_dict = OrderedDict(self.data)
            label_dict = OrderedDict(self.label)
            for k, _ in self.data:
                data_dict[k] = data_dict[k][:new_n]
            for k, _ in self.label:
                label_dict[k] = label_dict[k][:new_n]
            self.data = data_dict.items()
            self.label = label_dict.items()
        self.num_data = self.data_list[0].shape[0]
        assert self.num_data >= batch_size, \
            "batch_size need to be smaller than data size."
        self.cursor = -batch_size
        
    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.label]

    def hard_reset(self):
        """Igore roll over data and set to start"""
        self.cursor = -self.batch_size

    def reset(self):
        self.gen_idx = 0
        if self.last_batch_handle == 'roll_over' and self.cursor > self.num_data:
            self.cursor = -self.batch_size + (self.cursor%self.num_data)%self.batch_size
        else:
            self.cursor = -self.batch_size

    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor < self.num_data:
            return True
        else:
            return False

    def _generate_next(self):
        if (self.gen_idx>=self.gen_count):
            raise StopIteration()
        self.gen_idx += 1
        data, label = next(self.generator)
        self._reinit(data, label, self.batch_size, self.shuffle, self.last_batch_handle)
        
    def next(self):
        if (self.regenerate):
            self.regenerate = False
            self._generate_next() # This can raise StopIteration
        if self.iter_next():
            return DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=None)
        else:
            self.regenerate = True
            return self.next()  
            
    def _getdata(self, data_source):
        """Load data from underlying arrays, internal use only"""
        assert(self.cursor < self.num_data), "DataIter needs reset."
        if self.cursor + self.batch_size <= self.num_data:
            return [array(x[1][self.cursor:self.cursor+self.batch_size]) for x in data_source]
        else:
            pad = self.batch_size - self.num_data + self.cursor
            return [array(np.concatenate((x[1][self.cursor:], x[1][:pad]),
                                         axis=0)) for x in data_source]

    def getdata(self):
        return self._getdata(self.data)

    def getlabel(self):
        return self._getdata(self.label)

    def getpad(self):
        if self.last_batch_handle == 'pad' and \
           self.cursor + self.batch_size > self.num_data:
            return self.cursor + self.batch_size - self.num_data
        else:
            return 0
