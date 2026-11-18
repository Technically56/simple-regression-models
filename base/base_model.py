from abc import ABC, abstractmethod

import numpy as np

from base.base_backend import Array, BaseBackend

type Params = dict[str, Array]


class BaseModel(ABC):
    def __init__(self, backend: BaseBackend, n_features: int, l2: float = 0.0) -> None:
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        self.l2 = float(l2)
        self.n_features = n_features
        self.be = backend
        self._params: Params = {
            "w": self.be.zeros((n_features,)),
            "b": self.be.zeros(()),
        }

    @property
    def w(self) -> Array:
        return self._params["w"]

    @property
    def b(self) -> Array:
        return self._params["b"]

    @abstractmethod
    def forward(self, x: Array) -> Array:
        """(n, d) -> (n,). Linear: X·w + b. Logistic: sigmoid(X·w + b)."""

    def predict(self, x: Array, threshold: float = 0.5) -> Array:
        return self.forward(x) >= threshold

    @abstractmethod
    def cost(self, h: Array, y: Array) -> Array:
        """0-d device array: mean loss over the batch, without the L2 term."""

    def gradient(self, x: Array, y: Array) -> tuple[Array, Params]:
        h = self.forward(x)
        r = self.be.subtract(h, y)
        w = self._params["w"]
        dw = self.be.add(
            self.be.multiply(self.be.matmul(r, x), 1.0 / x.shape[0]),
            self.be.multiply(w, self.l2),
        )
        J = self.be.add(
            self.cost(h, y), self.be.multiply(self.be.matmul(w, w), 0.5 * self.l2)
        )
        return J, {"w": dw, "b": self.be.mean(r)}

    def step(self, grads: Params, lr: float) -> None:
        for name, p in self._params.items():
            self.be.axpy_(-lr, grads[name], p)

    def state(self) -> dict[str, np.ndarray]:
        return {k: self.be.to_host(p).copy() for k, p in self._params.items()}

    def load(self, state: dict[str, np.ndarray]) -> None:
        for k, p in self._params.items():
            if state[k].shape != p.shape:
                raise ValueError(f"{k}: expected {p.shape}, got {state[k].shape}")
        self._params = {k: self.be.to_device(np.array(state[k])) for k in self._params}

    def loss(self, x: Array, y: Array) -> Array:
        # Return the loss (cost + L2 term) for a batch of data.
        w = self._params["w"]
        return self.be.add(
            self.cost(self.forward(x), y),
            self.be.multiply(self.be.matmul(w, w), 0.5 * self.l2),
        )
