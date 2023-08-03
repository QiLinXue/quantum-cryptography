import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

class QKDSystem2Decoy:
    def __init__(self, e_detector=0.033, Y0=1.7e-6, eta_bob=0.045, e_0=0.5, alpha=0.21, N=2):
        self.e_detector = e_detector
        self.Y0 = Y0
        self.eta_bob = eta_bob
        self.e_0 = e_0
        self.alpha = alpha
        self.N = N

    def eta(self, L):
        return 10 ** (-self.alpha * L / 10) * self.eta_bob

    def Q(self, mu, L):
        return self.Y0 + 1 - np.exp(-self.eta(L) * mu)

    def QBER(self, mu, L):
        return self.e_0 * self.Y0 + self.e_detector * (1 - np.exp(-self.eta(L) * mu))

    def compute_Yi(self, mu, nu1, nu2, L):
        b_eq = [
            self.Q(mu, L) * np.exp(mu) - self.Y0,
            self.Q(nu1, L) * np.exp(nu1) - self.Y0,
            self.Q(nu2, L) * np.exp(nu2) - self.Y0,
        ]
        
        coefficients_mu = np.zeros(self.N)
        coefficients_nu1 = np.zeros(self.N)
        coefficients_nu2 = np.zeros(self.N)

        for i in range(1, self.N + 1):
            coefficients_mu[i - 1] = mu ** i / np.math.factorial(i)
            coefficients_nu1[i - 1] = nu1 ** i / np.math.factorial(i)
            coefficients_nu2[i - 1] = nu2 ** i / np.math.factorial(i)

        A_eq = [coefficients_mu, coefficients_nu1, coefficients_nu2]

        c = np.zeros(self.N)
        c[0] = 1

        # Solve the linear programming problem
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
        return result.x

    def compute_ei(self, mu, nu1, nu2, L):
        Yi = self.compute_Yi(mu, nu1, nu2, L)

        b_eq = [
            self.QBER(mu, L) * np.exp(mu) - self.e_0 * self.Y0,
            self.QBER(nu1, L) * np.exp(nu1) - self.e_0 * self.Y0,
            self.QBER(nu2, L) * np.exp(nu2) - self.e_0 * self.Y0,
        ]
        
        coefficients_mu = np.zeros(self.N)
        coefficients_nu1 = np.zeros(self.N)
        coefficients_nu2 = np.zeros(self.N)
        
        for i in range(1, self.N + 1):
            coefficients_mu[i - 1] = mu ** i / np.math.factorial(i) * Yi[i - 1]
            coefficients_nu1[i - 1] = nu1 ** i / np.math.factorial(i) * Yi[i - 1]
            coefficients_nu2[i - 1] = nu2 ** i / np.math.factorial(i) * Yi[i - 1]
            
        A_eq = [coefficients_mu, coefficients_nu1, coefficients_nu2]

        c = np.zeros(self.N)
        c[0] = -1

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 0.99))
        return Yi[0], result.x[0]

    @staticmethod
    def H2(e1):
        return -e1 * np.log2(e1) - (1 - e1) * np.log2(1 - e1)

    def compute_R(self, mu, nu1, nu2, L):
        Y1, e1 = self.compute_ei(mu, nu1, nu2, L)
        Q1 = Y1 * mu * np.exp(-mu)
        Q_mu = self.Y0 + 1 - np.exp(-self.eta(L) * mu)
        return -Q_mu * 1 * self.H2(self.e_detector) + Q1 * (1 - self.H2(e1))

    def get_R_vs_mu(self, mu_list, nu1, nu2, L):
        R_list = [self.compute_R(mu, nu1, nu2, L) for mu in mu_list]

        return R_list

qkd_system_2_decoy = QKDSystem2Decoy(N=100)

mu_list = np.linspace(0, 1, 100)
nu1 = 0.05
nu2 = 0
L = 160
R_list = qkd_system_2_decoy.get_R_vs_mu(mu_list, nu1, nu2, L)

plt.plot(mu_list, R_list)
plt.xlabel("mu")
plt.ylabel("R")

# Get the optimal value of mu
mu_opt = mu_list[np.argmax(R_list)]
print("mu_opt = ", mu_opt)

# Save plot in HD
plt.show()