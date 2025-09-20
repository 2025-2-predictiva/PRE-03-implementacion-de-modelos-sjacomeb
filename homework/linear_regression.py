import numpy as np
from tqdm import tqdm ##tqdm es una libreria que permite ver el progreso de un bucle for 


class LinearRegression:
    """Basic linear regression implementation."""

    def __init__(self, num_epochs=100, learning_rate=0.01):
        self.coefs_ = None
        self.intercept_ = None
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def fit(self, X, y): #X la matriz de coeficientes y y el vector de resultados
        """Fit the model using X, y."""

        self.coefs_ = np.random.rand(X.shape[1])
        self.intercept_ = np.random.rand()

        #tqdm permite ver la barrita del progreso de las iteraciones
        for _ in tqdm(range(self.num_epochs)): #_ es una variable que contiene el ultimo valor computado pero que no se usa luego en el codigo

            # Compute the predictions
            y_pred = np.matmul(X, self.coefs_) + self.intercept_

            # Compute the gradient of the intercept
            gradient_intercept_per_row = np.sum(-2 * (y - y_pred))
            gradient_intercept_ = np.sum(gradient_intercept_per_row) 

            # Compute the gradient of the coefficients
            gradient_coefs_per_row = -2 * np.matmul((y - y_pred), X)
            gradient_coefs_ = np.sum(gradient_coefs_per_row, axis=0)

            # Update the coefficients
            self.coefs_ -= self.learning_rate * gradient_coefs_
            self.intercept_ -= self.learning_rate * gradient_intercept_
            
    def predict(self, X):
        """Predict the target for the provided data."""
        return np.matmul(X, self.coefs_) + self.intercept_      