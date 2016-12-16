# pylint: disable=fixme, invalid-name, unused-argument, too-many-arguments, no-name-in-module
"""Common Optimization algorithms with regularizations."""
import math
import ctypes
from mxnet.base import _LIB, check_call
from mxnet.base import c_array, mx_uint, mx_float, c_str
from mxnet.base import OptimizerHandle, OptimizerCreator
from mxnet.ndarray import NDArray, zeros, clip, sqrt
from mxnet.random import normal
from mxnet.optimizer import *
import logging

@register
class MyAdaGrad(Optimizer):
    """AdaGrad optimizer of Duchi et al., 2011,

    This code follows the version in http://arxiv.org/pdf/1212.5701v1.pdf  Eq(5)
    by Matthew D. Zeiler, 2012. AdaGrad will help the network to converge faster
    in some cases.

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.05.

    wd : float, optional
        L2 regularization coefficient add to all the weights

    rescale_grad : float, optional
        rescaling factor of gradient.

    eps: float, optional
        A small float number to make the updating processing stable
        Default value is set to 1e-7.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
    """
    def __init__(self, eps=1e-7, **kwargs):
        super(MyAdaGrad, self).__init__(**kwargs)
        self.float_stable_eps = eps

    def create_state(self, index, weight):
        logging.debug("Creating Optimizer State")
        return zeros(weight.shape, weight.context) # history

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        lr = self._get_lr(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)
        history = state
        history[:] += (grad * grad)
        weight[:] += -lr * (grad / sqrt(history + self.float_stable_eps) + self.wd * weight)

