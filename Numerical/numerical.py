from __future__ import absolute_import, division, print_function, unicode_literals
from functools import reduce, partial, update_wrapper
import numpy as np
import itertools as it
import pandas as pd
import torch as th
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as G
from copy import deepcopy
from collections import Iterable, OrderedDict, namedtuple
import random
import math
from PIL import Image
import os
import warnings
warnings.filterwarnings("ignore")

class Torch(object):
	'''
	Torch class for wrapping the tensor object.

	Parameters
	----------
	tensor : torch.Tensor
		The tensor used for this Torch.
	'''
	def __init__(self, tensor):
		self._tensor = tensor

	@property
	def tensor(self):
		return self._tensor

	@classmethod
	def set_functional(cls, func):
		'''
		Create a Torch with the given function.

		Parameters
		----------
		func : function (with 0 input)
			The function used to create a Torch.

		Returns
		-------
		Torch : numerical.Torch
		'''
		return cls(func)

	@classmethod
	def set_embedding(cls, params):
		'''
		Create a Torch with the given parameters.

		Parameters
		----------
		params : torch.Tensor, numpy ndarray or list
		The parameters used to create a Torch.

		Returns
		-------
		Torch : numerical.Torch
		'''
		return cls(th.tensor(params, dtype=th.float32))

	@classmethod
	def set_tensor(cls, tensor):
		'''
		Create a Torch with the given tensor.

		Parameters
		----------
		tensor : torch.Tensor
		The tensor used to create a Torch.

		Returns
		-------
		Torch : numerical.Torch
		'''
		return cls(tensor.clone())

	@classmethod
	def random_tensor(cls, shape, dtype='float'):
		'''
		Create a Torch with a random tensor.

		Parameters
		----------
		shape : tuple of ints or int (optional)
            The shape of the tensor used to create a Torch. Default is None.
        dtype : str (optional)
            The type of the random values used to create a Torch. Default is 'float'.

		Returns
		-------
		Torch : numerical.Torch
		'''
		if dtype == 'float':
			return cls(th.randn(shape, dtype=th.float32))
		elif dtype in ['int', 'long']:
			return cls(th.zeros(shape, dtype=th.long))

	def to_device(self, device):
		self._tensor = self._tensor.to(device)

	def to_cpu(self):
		self._tensor = self._tensor.cpu()

	def to_numpy(self):
		return self._tensor.detach().clone().numpy().copy()

	def to_list(self):
		return self._tensor.tolist()

	def reshape(self, shape):
		self._tensor = self._tensor.reshape(shape)

	def expand_dim(self, dim):
		self._tensor = self._tensor.unsqueeze(dim)

	def squeeze(self, dim=None):
		self._tensor = self._tensor.squeeze(dim)

	def transpose(self, dim1, dim2):
		self._tensor = self._tensor.transpose(dim1, dim2)

	def gather(self, dim, index):
		self._tensor = self._tensor.gather(dim, index)

	def mask(self, mask):
		self._tensor = self._tensor.masked_fill(mask == 0, 0)

	def get_shape(self):
		return self._tensor.shape

	def get_dtype(self):
		return self._tensor.dtype

	def get_device(self):
		return self._tensor.device

	def detach(self):
		self._tensor = self._tensor.detach()

	def clone(self):
		return Torch(self._tensor.clone())

	def copy(self):
		return Torch(self._tensor.clone())

	def __iter__(self):
		for i in range(len(self._tensor)):
			yield self._tensor[i]

	def __len__(self):
		return len(self._tensor)

	def __getitem__(self, key):
		return self._tensor[key]

	def __setitem__(self, key, value):
		self._tensor[key] = value

	def __add__(self, other):
		if isinstance(other, Torch):
			return Torch(self._tensor + other.tensor)
		else:
			return Torch(self._tensor + other)

	def __radd__(self, other):
		if isinstance(other, Torch):
			return Torch(self._tensor + other.tensor)
		else:
			return Torch(self._tensor + other)

	def __sub__(self, other):
		if isinstance(other, Torch):
			return Torch(self._tensor - other.tensor)
		else:
			return Torch(self._tensor - other)

	def __rsub__(self, other):
		if isinstance(other, Torch):
			return Torch(self._tensor - other.tensor)
		else:
			return Torch(self._tensor - other)

	def __mul__(self, other):
		if isinstance(other, Torch):
			return Torch(self._tensor * other.tensor)
		else:
			return Torch(self._tensor * other)

	def __rmul__(self, other):
		if isinstance(other, Torch):
			return Torch(self._tensor * other.tensor)
		else:
			return Torch(self._tensor * other)

	def __truediv__(self, other):
		if isinstance(other, Torch):
			return Torch(self._tensor / other.tensor)
		else:
			return Torch(self._tensor / other)

	def __rtruediv__(self, other):
		if isinstance(other, Torch):
			return Torch(self._tensor / other.tensor)
		else:
			return Torch(self._tensor / other)

	def __pow__(self, other):
		if isinstance(other, Torch):
			return Torch(self._tensor ** other.tensor)
		else:
			return Torch(self._tensor ** other)

	def __rpow__(self, other):
		if isinstance(other, Torch):
			return Torch(self._tensor ** other.tensor)
		else:
			return Torch(self._tensor ** other)

	def __mod__(self, other):
		if isinstance(other, Torch):
			return Torch(self._tensor % other.tensor)
		else:
			return Torch(self._tensor % other)

	def __rmod__(self, other):
		if isinstance(other, Torch):
			return Torch(self._tensor % other.tensor)
		else:
			return Torch(self._tensor % other)

	def __neg__(self):
		return Torch(-self._tensor)

	def __iadd__(self, other):
		if isinstance(other, Torch):
			self._tensor += other.tensor
		else:
			self._tensor += other
		return self

	def __isub__(self, other):
		if isinstance(other, Torch):
			self._tensor -= other.tensor
		else:
			self._tensor -= other
		return self

	def __imul__(self, other):
		if isinstance(other, Torch):
			self._tensor *= other.tensor
		else:
			self._tensor *= other
		return self

	def __itruediv__(self, other):
		if isinstance(other, Torch):
			self._tensor /= other.tensor
		else:
			self._tensor /= other
		return self

	def __ipow__(self, other):
		if isinstance(other, Torch):
			self._tensor **= other.tensor
		else:
			self._tensor **= other
		return self

	def __imod__(self, other):
		if isinstance(other, Torch):
			self._tensor %= other.tensor
		else:
			self._tensor %= other
		return self

	def __call__(self, *args, **kwargs):
		return self._tensor(*args, **kwargs)

	def __repr__(self):
	    return self._tensor.__repr__()


class Group(object):
    '''
    Group class for wrapping the attribute object.

    Parameters
    ----------
    name : str
        The name of this Group. Default is None.
    attributes : list (optional)
        The list of the attributes used for this Group. Default is [].
    '''
    def __init__(self, name, *attributes):
        self._name = name
        if len(attributes) == 1 and len(attributes[0]) != 0:
            self._attributes = attributes[0]
        else:
            self._attributes = list(attributes)
        self._composed = False

    @property
    def name(self):
        return self._name

    @property
    def attributes(self):
        return self._attributes

    @property
    def composed(self):
        return self._composed

    def set_name(self, name):
        self._name = name

    def set_attribute(self, attribute):
        self._attributes = [attribute]

    def add_attribute(self, *attributes):
        self._attributes.extend(list(attributes))

    def to_device(self, device):
        for attribute in self._attributes:
            attribute.to_device(device)

    def to_cpu(self):
        for attribute in self._attributes:
            attribute.to_cpu()

    def to_numpy(self):
        results = []
        for attribute in self._attributes:
            if isinstance(attribute, Group):
                results.append(attribute.to_numpy())
            elif isinstance(attribute, Torch):
                results.append(attribute.to_numpy())
        return results

    def to_list(self):
        results = []
        for attribute in self._attributes:
            if isinstance(attribute, Group):
                results.append(attribute.to_list())
            elif isinstance(attribute, Torch):
                results.append(attribute.to_list())
        return results

    def get(self, name=None, index=None):
        if name and index:
            return [self[index] for self in self._attributes if self.name == name]
        if index is not None:
            if 0 <= index < len(self._attributes):
                return self._attributes[index]
            else:
                return [self[index] for self in self._attributes]
        if name is not None:
            return [self for self in self._attributes if self.name == name]

    def get_name(self, name=None, index=None):
        if name and index:
            return [self.name for self in self._attributes if self.name == name]
        if index is not None:
            if 0 <= index < len(self._attributes):
                return self._attributes[index].name
            else:
                return [self.name for self in self._attributes]
        if name is not None:
            return [self.name for self in self._attributes if self.name == name]

    def get_shape(self, name=None, index=None):
        if name and index:
            return [self.get_shape() for self in self._attributes if self.name == name]
        if index is not None:
            if 0 <= index < len(self._attributes):
                return self._attributes[index].get_shape()
            else:
                return [self.get_shape() for self in self._attributes]
        if name is not None:
            return [self.get_shape() for self in self._attributes if self.name == name]

    def get_dtype(self, name=None, index=None):
        if name and index:
            return [self.get_dtype() for self in self._attributes if self.name == name]
        if index is not None:
            if 0 <= index < len(self._attributes):
                return self._attributes[index].get_dtype()
            else:
                return [self.get_dtype() for self in self._attributes]
        if name is not None:
            return [self.get_dtype() for self in self._attributes if self.name == name]

    def get_device(self, name=None, index=None):
        if name and index:
            return [self.get_device() for self in self._attributes if self.name == name]
        if index is not None:
            if 0 <= index < len(self._attributes):
                return self._attributes[index].get_device()
            else:
                return [self.get_device() for self in self._attributes]
        if name is not None:
            return [self.get_device() for self in self._attributes if self.name == name]

    def detach(self, name=None, index=None):
        if name and index:
            for attribute in self._attributes:
                if attribute.name == name:
                    attribute[index].detach()
        if index is not None:
            if 0 <= index < len(self._attributes):
                self._attributes[index].detach()
            else:
                for attribute in self._attributes:
                    attribute[index].detach()
        if name is not None:
            for attribute in self._attributes:
                if attribute.name == name:
                    attribute.detach()

    def clone(self, name=None, index=None):
        results = []
        if name and index:
            for attribute in self._attributes:
                if attribute.name == name:
                    results.append(attribute[index].clone())
        if index is not None:
            if 0 <= index < len(self._attributes):
                results.append(self._attributes[index].clone())
            else:
                for attribute in self._attributes:
                    results.append(attribute[index].clone())
        if name is not None:
            for attribute in self._attributes:
                if attribute.name == name:
                    results.append(attribute.clone())
        if len(results) == 1:
            return results[0]
        return Group(self.name, results)

    def copy(self, name=None, index=None):
        results = []
        if name and index:
            for attribute in self._attributes:
                if attribute.name == name:
                    results.append(attribute[index].copy())
        if index is not None:
            if 0 <= index < len(self._attributes):
                results.append(self._attributes[index].copy())
            else:
                for attribute in self._attributes:
                    results.append(attribute[index].copy())
        if name is not None:
            for attribute in self._attributes:
                if attribute.name == name:
                    results.append(attribute.copy())
        if len(results) == 1:
            return results[0]
        return Group(self.name, results)

    def __iter__(self):
        for attribute in self._attributes:
            yield attribute

    def __len__(self):
        return len(self._attributes)

    def __getitem__(self, key):
        return self._attributes[key]

    def __setitem__(self, key, value):
        self._attributes[key] = value

    def __getattr__(self, attr):
        return lambda *args, **kwargs: [getattr(attribute, attr)(*args, **kwargs) for attribute in self._attributes]

    def __call__(self, *args, **kwargs):
        return [attribute(*args, **kwargs) for attribute in self._attributes]

    def __repr__(self):
        return self._attributes.__repr__()

    @classmethod
    def from_preset(cls, name):
        '''
        Create a Torch with a preset function.

        Parameters
        ----------
        name : str
            The name of the preset function used to create a Torch. Valid names are:
            'sigmoid', 'relu', 'tanh', 'softmax', 'linear', 'sequence'.

        Returns
        -------
        Torch : numerical.Torch
        '''
        if name == 'sigmoid':
            return Torch(th.sigmoid)
        elif name == 'relu':
            return Torch(th.relu)
        elif name == 'tanh':
            return Torch(th.tanh)
        elif name == 'softmax':
            return Torch(th.softmax)
        elif name == 'linear':
            return Torch(lambda x: 1 * x)
        elif name == 'sequence':
            return Torch(lambda x, dim=1: x.sum(dim))

    @classmethod
    def from_preset_with_parameters(cls, name, shape):
        '''
        Create a Torch with a preset function and parameters.

        Parameters
        ----------
        name : str
            The name of the preset function used to create a Torch. Valid names are:
            'sigmoid', 'relu', 'tanh', 'softmax', 'linear', 'sequence'.
        shape : tuple of ints or int
            The shape of the parameters used to create a Torch.

        Returns
        -------
        Torch : numerical.Torch
        '''
        if name == 'sigmoid':
            return Torch(th.sigmoid)
        elif name == 'relu':
            return Torch(th.relu)
        elif name == 'tanh':
            return Torch(th.tanh)
        elif name == 'softmax':
            return Torch(th.softmax)
        elif name == 'linear':
            return Torch.set_embedding(Torch.random_tensor(shape, 'float'))
        elif name == 'sequence':
            return Torch(lambda x, dim=1: x.sum(dim))

    def compose(self, *args, **kwargs):
        if not self._composed:
            self._compose(*args, **kwargs)
            self._composed = True
        else:
            raise Exception('This Group has already been composed.')

    def _compose(self, *args, **kwargs):
        pass

    def compile(self):
        if not self._composed:
            raise Exception('Please compose this Group first.')
        else:
            self._compile()

    def _compile(self):
        pass

    def auto_compose(self, out_projection_type='logits', resizing=True):
        if not self._composed:
            self._auto_compose(out_projection_type, resizing)
            self._composed = True
        else:
            raise Exception('This Group has already been composed.')

    def _auto_compose(self, out_projection_type='logits', resizing=True):
        for i in range(len(self._attributes)):
            attribute = self._attributes[i]
            if isinstance(attribute, Group) and not attribute.composed:
                attribute.auto_compose(out_projection_type, resizing)
        for i in range(len(self._attributes) - 1):
            attribute = self._attributes[i]
            if isinstance(attribute, Group) and not attribute.composed:
                shape = reduce(lambda x, y: x * y, attribute.get_shape()[1:])
                self._attributes[i + 1].add_attribute(Torch.set_embedding(Torch.random_tensor((shape,), 'float')))
        if isinstance(self._attributes[-1], Group):
            shape = reduce(lambda x, y: x * y, self._attributes[-1].get_shape()[1:])
            if out_projection_type == 'logits':
                self.add_attribute(Torch.set_embedding(Torch.random_tensor((shape,), 'float')))
            elif out_projection_type == 'softmax':
                self.add_attribute(Torch.set_embedding(Torch.random_tensor((shape,), 'float')),
                                   Torch.set_functional(th.softmax))
        else:
            if out_projection_type == 'logits':
                pass
            elif out_projection_type == 'softmax':
                self.add_attribute(Torch.set_functional(th.softmax))

    def auto_compile(self, out_projection_type='logits', resizing=True):
        if not self._composed:
            raise Exception('Please compose this Group first.')
        else:
            self._auto_compile(out_projection_type, resizing)

    def _auto_compile(self, out_projection_type='logits', resizing=True):
        for i in range(len(self._attributes)):
            attribute = self._attributes[i]
            if isinstance(attribute, Group) and not attribute.composed:
                attribute.auto_compile(out_projection_type, resizing)
        for i in range(len(self._attributes) - 1):
            attribute = self._attributes[i]
            if isinstance(attribute, Group) and not attribute.composed:
                shape = reduce(lambda x, y: x * y, attribute.get_shape()[1:])
                self._attributes[i + 1].add_attribute(Torch.set_embedding(Torch.random_tensor((shape,), 'float')))
        if isinstance(self._attributes[-1], Group):
            shape = reduce(lambda x, y: x * y, self._attributes[-1].get_shape()[1:])
            if out_projection_type == 'logits':
                self.add_attribute(Torch.set_embedding(Torch.random_tensor((shape,), 'float')))
            elif out_projection_type == 'softmax':
                self.add_attribute(Torch.set_embedding(Torch.random_tensor((shape,), 'float')),
                                   Torch.set_functional(th.softmax))
        else:
            if out_projection_type == 'logits':
                pass
            elif out_projection_type == 'softmax':
                self.add_attribute(Torch.set_functional(th.softmax))
        if resizing:
            for i in range(len(self._attributes) - 1):
                shape = reduce(lambda x, y: x * y, self._attributes[i].get_shape()[1:])
                self._attributes[i + 1].add_attribute(Torch.set_embedding(Torch.random_tensor((shape,), 'float')))
            if isinstance(self._attributes[-1], Group):
                shape = reduce(lambda x, y: x * y, self._attributes[-1].get_shape()[1:])
                if out_projection_type == 'logits':
                    self.add_attribute(Torch.set_embedding(Torch.random_tensor((shape,), 'float')))
                elif out_projection_type == 'softmax':
                    self.add_attribute(Torch.set_embedding(Torch.random_tensor((shape,), 'float')),
                                       Torch.set_functional(th.softmax))
            else:
                if out_projection_type == 'logits':
                    pass
                elif out_projection_type == 'softmax':
                    self.add_attribute(Torch.set_functional(th.softmax))


class Model(Group):
    '''
    Model class for wrapping the attribute object.

    Parameters
    ----------
    name : str
        The name of this Model. Default is None.
    attributes : list (optional)
        The list of the attributes used for this Model. Default is [].
    '''
    def __init__(self, name, *attributes):
        super(Model, self).__init__(name, *attributes)

    def _compile(self):
        for attribute in self._attributes:
            if isinstance(attribute, Group):
                attribute.compile()

    def train(self):
        for attribute in self._attributes:
            if isinstance(attribute, Group):
                attribute.train()

    def eval(self):
        for attribute in self._attributes:
            if isinstance(attribute, Group):
                attribute.eval()


class TrainableModel(Model):
    '''
    TrainableModel class for wrapping the attribute object.

    Parameters
    ----------
    name : str
        The name of this TrainableModel. Default is None.
    attributes : list (optional)
        The list of the attributes used for this TrainableModel. Default is [].
    '''
    def __init__(self, name, *attributes):
        super(TrainableModel, self).__init__(name, *attributes)

    def _compose(self, loss_function=None, optimizer=None, scheduler=None, metrics=None):
        if loss_function:
            self.add_attribute(Torch.set_functional(loss_function))
        if optimizer:
            self.add_attribute(optimizer)
        if scheduler:
            self.add_attribute(scheduler)
        if metrics:
            for metric in metrics:
                self.add_attribute(metric)

    def _compile(self):
        super(TrainableModel, self)._compile()
        for attribute in self._attributes:
            if isinstance(attribute, th.optim.Optimizer):
                self._optimizer = attribute
            elif isinstance(attribute, _LRScheduler):
                self._scheduler = attribute

    def train(self, mode=True):
        super(TrainableModel, self).train()
        if mode:
            self._optimizer.zero_grad()
        else:
            with th.no_grad():
                self._scheduler.step()

    def eval(self, mode=True):
        super(TrainableModel, self).eval()
        if mode:
            with th.no_grad():
                self._scheduler.step()

    def _get_loss(self):
        return self._attributes[0]

    def _get_optimizer(self):
        return self._optimizer

    def _get_scheduler(self):
        return self._scheduler

    def _get_metrics(self):
        return [attribute for attribute in self._attributes[3:]]


class DataSet(Dataset):
    '''
    Dataset class for wrapping the data object.

    Parameters
    ----------
    data : tuple, list or numpy array
        The data used for this DataSet.
    '''
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)


class DataLoader(DataLoader):
    '''
    DataLoader class for wrapping the dataset and batch_size.

    Parameters
    ----------
    dataset : numerical.DataSet
        The dataset used for this DataLoader.
    batch_size : int or None (optional)
        The batch size used for this DataLoader. Default is 1.

    Returns
    -------
    torch.utils.data.DataLoader : torch.utils.data.DataLoader
        The DataLoader object.
    '''
    def __init__(self, dataset, batch_size=1):
        super(DataLoader, self).__init__(dataset, batch_size=batch_size)


def _batching(iterator, batch_size):
    batch = []
    for element in iterator:
        batch.append(element)
        if len(batch) == batch_size:
            yield tuple(list(x) for x in zip(*batch))
            batch = []
    if batch:
        yield tuple(list(x) for x in zip(*batch))


class TensorDataset(DataSet):
    '''
    TensorDataset class for wrapping the data object.

    Parameters
    ----------
    *tensors : torch.Tensor or numerical.Torch (optional)
        The tensors used for this TensorDataset. Default is ().

    Returns
    -------
    numerical.DataSet : numerical.DataSet
        The DataSet object.
    '''
    def __init__(self, *tensors):
        super(TensorDataset, self).__init__([tensor.tensor for tensor in tensors])


class TensorDataLoader(DataLoader):
    '''
    TensorDataLoader class for wrapping the dataset and batch_size.

    Parameters
    ----------
    *tensors : torch.Tensor or numerical.Torch (optional)
        The tensors used for this TensorDataLoader. Default is ().
    batch_size : int or None (optional)
        The batch size used for this TensorDataLoader. Default is 1.

    Returns
    -------
    torch.utils.data.DataLoader : torch.utils.data.DataLoader
        The DataLoader object.
    '''
    def __init__(self, *tensors, **kwargs):
        super(TensorDataLoader, self).__init__(TensorDataset(*tensors), **kwargs)
