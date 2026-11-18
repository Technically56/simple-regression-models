from base.base_backend import Array
from base.base_model import BaseModel


class LinearRegressionModel(BaseModel):
    """Linear regression model with L2 regularization."""

    def forward(self, x: Array) -> Array:
        return self.be.add(self.be.matmul(x, self.w), self.b)

    def cost(self, h: Array, y: Array) -> Array:
        diff = self.be.subtract(h, y)
        return self.be.multiply(self.be.mean(self.be.multiply(diff, diff)), 0.5)


class LogisticRegressionModel(BaseModel):
    """Logistic regression model with L2 regularization."""

    def forward(self, x: Array) -> Array:
        z = self.be.add(self.be.matmul(x, self.w), self.b)
        return self.be.sigmoid(z)

    def cost(self, h: Array, y: Array) -> Array:
        eps = 1e-7
        h_clipped = self.be.clip(h, eps, 1 - eps)
        term1 = self.be.multiply(y, self.be.log(h_clipped))
        term2 = self.be.multiply(
            self.be.subtract(1.0, y), self.be.log(self.be.subtract(1.0, h_clipped))
        )
        return self.be.multiply(self.be.mean(self.be.subtract(-term1, term2)), 1.0)
