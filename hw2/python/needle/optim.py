"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is not None:
                if param not in self.u:
                    self.u[param] = ndl.zeros_like(param.grad, requires_grad=False)
                self.u[param] = self.momentum * self.u[param].data + (1 - self.momentum) * (param.grad.data + self.weight_decay * param.data)
                param.data = param.data - self.lr * self.u[param]

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION



class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for i, param in enumerate(self.params):
            if i not in self.m:
                self.m[i] = ndl.init.zeros(*param.shape)
                self.v[i] = ndl.init.zeros(*param.shape)
            
            if param.grad is None:
                continue
            
            grad_data = param.grad
            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data
            self.m[i] = self.beta1 * self.m[i]+(1 - self.beta1) * grad_data.data
            self.v[i] = self.beta2 * self.v[i]+(1 - self.beta2) * grad_data**2
            # 修正
            u_hat = (self.m[i]) / (1 - self.beta1 ** self.t)
            v_hat = (self.v[i]) / (1 - self.beta2 ** self.t)
            param.data -= self.lr * u_hat.data /(ndl.ops.power_scalar(v_hat.data, 0.5) + self.eps).data
        ### END YOUR SOLUTION



