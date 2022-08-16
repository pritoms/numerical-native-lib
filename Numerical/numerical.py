from abc import ABCMeta, abstractmethod
import torch
import numpy as np
from copy import deepcopy
from collections import OrderedDict

class Torch:
    def __init__(self, name=None, tensor=None):
        self.name = name
        self.tensor = tensor
        self.shape = None if self.tensor is None else self.tensor.shape

    def __getattr__(self, item):
        return getattr(self.tensor, item)

    def __add__(self, torch_):
        if isinstance(torch_, Torch):
            torch_ = torch_.tensor
        return Torch(torch=self.tensor + torch_)

    def __sub__(self, torch_):
        if isinstance(torch_, Torch):
            torch_ = torch_.tensor
        return Torch(torch=self.tensor - torch_)

    def __mul__(self, torch_):
        if isinstance(torch_, Torch):
            torch_ = torch_.tensor
        return Torch(torch=self.tensor * torch_)

    def __truediv__(self, torch_):
        if isinstance(torch_, Torch):
            torch_ = torch_.tensor
        return Torch(torch=self.tensor / torch_)

    def __pow__(self, n):
        return Torch(torch=self.tensor ** n)

    def __repr__(self):
        return self.name + ': ' + str(self.shape)

    @staticmethod
    def random_tensor(shape, dtype='float'):
        tensor = torch.randn(*shape).to(dtype)
        return Torch(tensor=tensor)

    @staticmethod
    def parameterize_like(torch_, requires_grad=True):
        tensor = torch.nn.Parameter(deepcopy(torch_.tensor), requires_grad=requires_grad)
        return Torch(name=torch_.name, tensor=tensor)

    @staticmethod
    def set_embedding(torch_, requires_grad=True):
        tensor = torch.nn.Embedding(torch_.shape[0], torch_.shape[1])
        tensor.weight = torch.nn.Parameter(deepcopy(torch_.tensor), requires_grad=requires_grad)
        return Torch(name=torch_.name, tensor=tensor)

    @staticmethod
    def set_functional(torch_):
        if isinstance(torch_, Group):
            if torch_.is_composed:
                return torch_.composed
            else:
                raise ValueError('The group must be composed before applying functional operation.')
        else:
            return torch_


class Group:
    __metaclass__ = ABCMeta

    def __init__(self, name, *attributes):
        self.name = name
        self.attributes = attributes
        self.composed = None
        self.is_composed = False

    def __repr__(self):
        return self.name

    def __getitem__(self, item):
        return self.attributes[item]

    def __call__(self, *inputs):
        return self.composed(*inputs)

    def __len__(self):
        return len(self.attributes)

    def __iter__(self):
        return iter(self.attributes)

    def add_attribute(self, *attributes):
        self.attributes += attributes

    @abstractmethod
    def auto_compose(self):
        pass

    @staticmethod
    def from_preset(operation, name=None):
        if operation == 'relu':
            return ReLU(name=name)
        elif operation == 'linear':
            return Linear(name=name)
        else:
            raise ValueError('Unsupported operation.')

    @staticmethod
    def from_torch(torch_, name=None):
        if torch_.tensor is None:
            raise ValueError('A torch tensor must be given')
        else:
            return TorchGroup(name=name, torch=torch_)

    @staticmethod
    def from_numpy(numpy_, name=None):
        if numpy_ is None:
            raise ValueError('A numpy array must be given')
        else:
            return TorchGroup(name=name, torch=Torch.random_tensor(numpy_.shape))

    @staticmethod
    def from_torch_tensor(torch_tensor, name=None):
        if torch_tensor is None:
            raise ValueError('A torch tensor must be given')
        else:
            return TorchGroup(name=name, torch=torch_tensor)

    @staticmethod
    def from_numpy_array(numpy_array, name=None):
        if numpy_array is None:
            raise ValueError('A numpy array must be given')
        else:
            return TorchGroup(name=name, torch=Torch.random_tensor(numpy_array.shape))

    @staticmethod
    def compose(*operations):
        composed = torch.nn.Sequential(OrderedDict([(operation.name, operation) for operation in operations]))
        return composed


class TorchGroup(Group):
    def __init__(self, name, torch):
        self.name = name
        self.torch = torch
        self.composed = self.torch.tensor

    def auto_compose(self):
        return None

    def __repr__(self):
        return self.torch.__repr__()

    def __call__(self, *inputs):
        return self.torch(*inputs)

    def __len__(self):
        return len(self.torch)

    def __iter__(self):
        return iter(self.torch)


class ReLU(Group):
    def __init__(self, name=None):
        super().__init__(name=name)

    def auto_compose(self):
        self.composed = torch.nn.ReLU()
        self.is_composed = True

    def __call__(self, *inputs):
        return self.composed(*inputs)


class Linear(Group):
    def __init__(self, name=None):
        super().__init__(name=name)

    def auto_compose(self):
        self.composed = torch.nn.Linear(self.attributes[1].shape[0], self.attributes[0].shape[1])
        self.is_composed = True

    def __call__(self, *inputs):
        return self.composed(*inputs)


class Sequential(Group):
    def __init__(self, name=None):
        super().__init__(name=name)

    def auto_compose(self):
        operations = []
        for attribute in self.attributes:
            if isinstance(attribute, Group):
                operations += [attribute]
            elif isinstance(attribute, Torch):
                operations += [TorchGroup.from_torch(attribute)]
            else:
                raise ValueError('Unsupported attribute type.')

        # Compose groups and tensors into a sequential model.
        composed = torch.nn.Sequential(OrderedDict([(operation.name, operation) for operation in operations]))
        self.composed = composed
        self.is_composed = True

    def __call__(self, *inputs):
        return self.composed(*inputs)


class Parallel(Group):
    def __init__(self, name=None):
        super().__init__(name=name)

    def auto_compose(self):
        operations = []
        for attribute in self.attributes:
            if isinstance(attribute, Group):
                operations += [attribute]
            elif isinstance(attribute, Torch):
                operations += [TorchGroup.from_torch(attribute)]
            else:
                raise ValueError('Unsupported attribute type.')

        # Compose groups and tensors into a sequential model.
        composed = torch.nn.ModuleList([operation for operation in operations])
        self.composed = composed
        self.is_composed = True

    def __call__(self, *inputs):
        return self.composed(*inputs)

    def __iter__(self):
        return iter(self.composed)

    def __len__(self):
        return len(self.composed)


class Model(Group):
    def __init__(self, name=None):
        super().__init__(name=name)

    def auto_compose(self):
        operations = []
        for attribute in self.attributes:
            if isinstance(attribute, Group):
                operations += [attribute]
            elif isinstance(attribute, Torch):
                operations += [TorchGroup.from_torch(attribute)]
            else:
                raise ValueError('Unsupported attribute type.')

        # Compose groups and tensors into a sequential model.
        composed = torch.nn.ModuleList([operation for operation in operations])
        self.composed = composed
        self.is_composed = True

    def compile(self):
        self.composed = torch.nn.Sequential(OrderedDict([(operation.name, operation) for operation in self]))

    def fit(self, X, y, optimizer=None, loss_function=None, epochs=1):
        if not optimizer:
            optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        if not loss_function:
            loss_function = torch.nn.MSELoss()

        # Train the model
        for epoch in range(epochs):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self(X)

            # Compute and print loss
            loss = loss_function(y_pred, y)
            print(f'Epoch {epoch + 1}: Loss {loss.item():.4f}')

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, X):
        return self(X)

    def __iter__(self):
        return iter(self.composed)

    def __len__(self):
        return len(self.composed)

    def __call__(self, *inputs):
        return self.composed(*inputs)
