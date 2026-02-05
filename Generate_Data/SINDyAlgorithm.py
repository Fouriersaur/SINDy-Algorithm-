import numpy as np
import LorentzAttractor

lorenz_data = LorentzAttractor.LorentzAttractor()

# Extract the data
x = lorenz_data.x
y = lorenz_data.y
z = lorenz_data.z
t_eval = lorenz_data.t_eval

# Compute derivatives
dxdt = np.gradient(x, t_eval)
dydt = np.gradient(y, t_eval)
dzdt = np.gradient(z, t_eval)

dXdt = np.vstack((dxdt, dydt, dzdt)).T

# Verify everything
print(dxdt[:5], dydt[:5], dzdt[:5])  # Print first 5 values of derivatives

# Library of candidate functions (polynomials up to degree 3)
def library(x, y, z):
    return np.array([
        np.ones_like(x),  # Constant term
        x, y, z,          # Linear terms
        x**2, y**2, z**2, # Quadratic terms
        x*y, x*z, y*z,    # Interaction terms
        x**3, y**3, z**3  # Cubic terms
    ]).T
# Construct the library
Theta = library(x, y, z)

print(Theta.shape[1])  # Print the shape of the library matrix

# Orindary Linear Regression to find the coefficients
Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]

max_iteration = 10

for iteration in range(max_iteration):
    for i in range(Xi.shape[1]):
        for k in range(Xi.shape[0]):
            if abs(Xi[k, i]) > 0.01:  # Threshold to consider as non-zero
                Xi[k, i] = Xi[k, i]  # Keep significant coefficients
                Theta[:, k] = Theta[:, k]  # Keep the corresponding term in the library
            else:
                Xi[k, i] = 0.0  # Set small coefficients to zero for sparsity
                Theta[:, k] = 0.0  # Remove the corresponding term from the library

    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]


print(Xi)  

