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

Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]
threshold = 0.01
max_iter = 10

for iteration in range(max_iter):
    Xi_old = Xi.copy()
    
    for i in range(Xi.shape[1]):  # loop over variables
        active_indices = np.where(abs(Xi[:, i]) > threshold)[0]
        if len(active_indices) == 0:
            continue
        
        # Solve least squares only for active terms
        Theta_active = Theta[:, active_indices]
        Xi_active = np.linalg.lstsq(Theta_active, dXdt[:, i], rcond=None)[0]
        
        # Update Xi
        Xi[:, i] = 0 # Reset all coefficients to zero before updating
        Xi[active_indices, i] = Xi_active
    
    # Stop if converged
    if np.allclose(Xi, Xi_old, atol=1e-6):
        break

print("Sparse Xi:")
print(Xi)  

