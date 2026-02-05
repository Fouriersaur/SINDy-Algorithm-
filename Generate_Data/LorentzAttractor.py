import numpy as np
from scipy.integrate import solve_ivp

class LorentzAttractor:
    def __init__(self, sigma=10.0, beta=8/3, rho=28.0, t_max=25.0, n_points=10000):
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        
        # Time array
        self.t_eval = np.linspace(0, t_max, n_points)
        
        # Initial conditions
        self.initial_state = [1.0, 1.0, 1.0]
        
        # Solve the Lorenz system
        sol = solve_ivp(self.lorenz, [0, t_max], self.initial_state, t_eval=self.t_eval)
        
        self.x = sol.y[0]
        self.y = sol.y[1]
        self.z = sol.y[2]

    def lorenz(self, t, state):
        x, y, z = state
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return [dxdt, dydt, dzdt]

if __name__ == "__main__":
    lorenz_data = LorentzAttractor()
    print("Lorenz data generated successfully")
