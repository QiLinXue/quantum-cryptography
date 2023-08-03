import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import asyncio

class Repeater():
    '''
    Base Repeater class with parameters and common methods
    '''
    alpha = 0.046 * (1 / 1000) # km^-1 (Fiber loss coefficient)
    beta = 0.62 # m^-1 (On-chip loss coefficient)
    tau_f = 102.85 * (1e-9) # ns (feed-forward time in fiber)
    tau_s = 20 * (1e-12) # ps (feed-forward time on chip)
    eta_c = 0.99 # chip to fiber coupling efficiency
    eta_s = 0.99 # source detector efficient product
    c_f = 2e8 # m/s (speed of light in fiber)
    c_ch = 7.6e7 # m/s (speed of light in chip)
    eta_GHZ = eta_s/(2-eta_s)
    P_GHZ = (eta_s*(2-eta_s))**3/32
    P_chip = np.exp(-beta*tau_s*c_ch)
    P_fib = np.exp(-alpha*tau_f*c_f)

    def __init__(self, k=7, m=4, n=250):
        self.k = k
        self.m = m
        self.n = n
        self.P_prime = (self.eta_GHZ*self.P_chip**(self.k+1))**(4*self.m+1)

class NaiveRepeater(Repeater):
    '''
    Naive Repeater subclass with naive approach for computing P_cn
    '''
    def Ql(self, l):
        return (self.eta_GHZ*self.P_chip**l)**2/2

    def P_c1(self, n_meas, n_B, n_GHZ):
        return 1 - (1 - self.P_prime * self.Pl(self.k, n_B, n_GHZ))**n_meas

    def get_P_c1_from_Ns(self, Ns):
        n_meas_list = np.linspace(1, 50, 50)
        n_B_list = np.linspace(1, 50, 50)

        optimal_P_c1 = -1
        for n_meas in n_meas_list:
            for n_B in n_B_list:
                n_GHZ = int(self.get_n_GHZ(Ns, n_meas, n_B))
                if n_GHZ < 1:
                    continue
                P_c1_current = self.P_c1(n_meas, n_B, n_GHZ)
                if P_c1_current > optimal_P_c1:
                    optimal_P_c1 = P_c1_current
        return optimal_P_c1

    def get_P_cn_from_Ns(self, Ns):
        return self.get_P_c1_from_Ns(Ns)**self.n

    def P0(self, n_GHZ):
        return 1 - (1 - self.P_GHZ)**n_GHZ

    def Pl(self, l, n_B, n_GHZ):
        if l == 0:
            return self.P0(n_GHZ)
        return 1 - (1 - self.Pl(l-1, n_B, n_GHZ)**2 * self.Ql(l))**n_B

    def get_n_GHZ(self, Ns, n_meas, n_B):
        return Ns/(6*n_meas*(2*n_B)**self.k)

    def get_rate(self, b, L, Ns):
        if Ns == None:
            P_cn = 0.9
        else:
            P_cn = self.get_P_cn_from_Ns(Ns)

        A = self.m*(0.5*self.eta_s**2 + 0.25*self.eta_s**4)
        B = self.P_chip**(self.k+2)*self.eta_GHZ*self.eta_c
        C = self.P_chip**(self.k+2)*self.eta_GHZ*self.eta_c
        # eta = ((A*B**2)**(-z))**n
        eta = np.exp(-self.alpha*L)

        P_X = 1 - (1-(eta**(1/(2*self.n)))**(b[1] + 1) * B**(b[1]+1))**b[0]
        P_Z = (1 - (1 - eta**(1/(2*self.n))*B)**(b[1]+1))**b[0]

        N = 2*self.m*(b[0]*b[1]+b[0]+1)
        P_B = A*B**2/self.m*eta**(1/(self.n))
        P_end = 1 - (1 - eta**(1/(2*self.n))*C)**(N/2)

        return P_cn/(N) * P_end**2 * P_Z**(2*self.n*(self.m-1)) * P_X**(2*self.n) * (1 - (1-P_B)**self.m)**(self.n-1)
    
    def get_min_Ns(self, Ns_min=6, Ns_max=16):
        Ns_list = np.logspace(Ns_min, Ns_max, 10)
        threshold = 0.9

        for Ns in Ns_list:
            if self.get_P_cn_from_Ns(Ns) > threshold:
                return Ns

class ImprovedRepeater(Repeater):
    '''
    Improved Repeater subclass with advanced approach for computing P_cn
    '''
    def pl(self, l):
        mu_l = self.eta_GHZ * self.P_chip**(l+1)
        return mu_l**2 * (0.5*self.eta_s**2 + 0.25*self.eta_s**4)

    def num_Clm_states(self, l, Ns, multi_index=[]):
        if l == 0:
            return np.random.binomial(Ns/6, self.P_GHZ)/(2**self.k)
        else:
            y1 = int(self.num_Clm_states(l-1, Ns, multi_index + [1]))
            y2 = int(self.num_Clm_states(l-1, Ns, multi_index + [2]))
            return np.random.binomial(min(y1, y2), self.pl(l))

    # async def worker(self, k, Ns):
    #     return self.num_Clm_states(k, Ns) > 0

    # async def P_c1_advanced_helper(self, Ns, total_sims=500):
    #     tasks = [self.worker(self.k, Ns) for _ in range(total_sims)]
    #     results = await asyncio.gather(*tasks)
    #     success = sum(results)
    #     return success / total_sims
    
    # def P_c1_advanced(self, Ns, total_sims=500):
    #     cur_time = time.time()
    #     result = asyncio.run(self.P_c1_advanced_helper(Ns, total_sims))
    #     print("Time taken: %s" % (time.time() - cur_time))
    #     return result
    def P_c1_advanced(self, Ns, total_sims = 1000):
        success = 0
        cur_time = time.time()

        for i in range(total_sims):
            if self.num_Clm_states(self.k, Ns) > 0:
                success += 1
        # print("Time taken: %s" % (time.time() - cur_time))

        return success / total_sims

    def get_P_cn_from_Ns(self, Ns, total_sims = 1000):
        return self.P_c1_advanced(Ns, total_sims)**self.n

    def get_rate(self, b, L, Ns, total_sims = 1000):
        if Ns == None:
            P_cn = 0.9
        else:
            P_cn = self.get_P_cn_from_Ns(Ns, total_sims)

        A = self.m*(0.5*self.eta_s**2 + 0.25*self.eta_s**4)/self.P_fib**2
        B = self.P_chip**(self.k+2)*self.P_fib*self.eta_GHZ*self.eta_c
        C = self.P_chip**(self.k+2)*self.eta_GHZ*self.eta_c
        # eta = ((A*B**2)**(-z))**n
        eta = np.exp(-self.alpha*L)

        P_X = 1 - (1-(eta**(1/self.n))**(b[1] + 1) * B**(b[1]+1))**b[0]
        P_Z = (1 - (1 - eta**(1/self.n)*B)**(b[1]+1))**b[0]
        P_B = A*B**2/self.m*eta**(1/self.n)
        P_end = 1 - (1 - eta**(1/(2*self.n))*C)**self.m

        N = 2*self.m

        return P_cn/(N) * P_end**2 * P_Z**(2*self.n*(self.m-1)) * P_X**(2*self.n) * (1 - (1-P_B)**self.m)**(self.n-1)

    def get_min_Ns(self, Ns_min=6, Ns_max=12):
        Ns_list = np.logspace(Ns_min, Ns_max, 20)
        threshold = 0.9

        for Ns in Ns_list:
            if self.get_P_cn_from_Ns(Ns, 1000) > threshold:
                return Ns
        return Ns_list[-1]
def make_plot_1():
    # Instantiate the Repeater classes
    naive_repeater = NaiveRepeater()
    improved_repeater = ImprovedRepeater()

    naive_N_s_list = np.logspace(9, 12, 40)
    improved_N_s_list = np.logspace(6, 7, 20)

    # Compute P_cn for both types of Repeater
    start_time = time.time()
    naive_P_cn_list = [naive_repeater.get_P_cn_from_Ns(i) for i in naive_N_s_list]

    with mp.Pool() as pool:
        improved_P_cn_list = pool.map(improved_repeater.get_P_cn_from_Ns, improved_N_s_list)
        pool.close()

    elapsed_time = time.time() - start_time
    print(elapsed_time)

    cutoff_index = next((i for i, value in enumerate(naive_P_cn_list) if value > 1e-30)) - 1

    naive_N_s_list = naive_N_s_list[cutoff_index:]
    naive_P_cn_list = naive_P_cn_list[cutoff_index:]

    # Plot for Naive Repeater
    plt.figure(figsize=(10,7))
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(naive_N_s_list, naive_P_cn_list, label='Naive Repeater')
    plt.plot(improved_N_s_list, improved_P_cn_list, label='Improved Repeater')
    plt.xlim(1e5, 1e12)
    plt.ylim(1e-30, 1)
    plt.xticks([10**i for i in range(6, 14, 2)])
    plt.legend()
    plt.show()

def get_R_list(L_list, n):
    print("Computing R_list for n = %s" % n)
    repeater = ImprovedRepeater(k=8, n=n, m = 4)
    return [repeater.get_rate([7,3], L, 10**7) for L in L_list]

def make_plot_2():
    L_list = np.linspace(0, 1000e3, 100)
    R_list_1 = get_R_list(L_list, 1)
    R_list_2 = get_R_list(L_list, 10)
    R_list_3 = get_R_list(L_list, 24)
    R_list_4 = get_R_list(L_list, 56)
    R_list_5 = get_R_list(L_list, 133)
    R_list_6 = get_R_list(L_list, 314)

    # Plot with magenta dotted lines
    plt.plot(L_list, R_list_1, '--', color='magenta')
    plt.plot(L_list, R_list_2, '--', color='magenta')
    plt.plot(L_list, R_list_3, '--', color='magenta')
    plt.plot(L_list, R_list_4, '--', color='magenta')
    plt.plot(L_list, R_list_5, '--', color='magenta')
    plt.plot(L_list, R_list_6, '--', color='magenta')

    plt.xlabel("L (km)")
    plt.ylabel("Secret key bits per mode")
    plt.xlim(0, 1000e3)
    plt.ylim(1e-9, 1)
    plt.yscale("log")
    # plt.show()
    # Save plot
    plt.savefig("secret_key_bits_per_mode.png")

def get_optimal_rate(k, m, b0, b1, L, repeater_type):
    n_range = np.linspace(1, 3000, 200)
    n_range = [int(i) for i in n_range]
    rate_list = [repeater_type(k=k, m=m, n=n).get_rate([b0, b1], L, None) for n in n_range]
    optimal_n = n_range[np.argmax(rate_list)]
    optimal_rate = np.max(rate_list)
    # if optimal_n == n_range[0] or optimal_n == n_range[-1]:
        # print("Warning: optimal n is at the edge of n_range")
    return optimal_rate, optimal_n

def get_optimal_params(k, repeater_type, m_aim, b0_aim, b1_aim, L = 300e3):
    # Optimal rate
    rate_opt = None
    b_opt = [None, None]
    m_opt = None
    n_opt = None

    threshold = 3
    m_min = max(m_aim - threshold, 1)
    m_max = m_aim + threshold
    b0_min = max(b0_aim - threshold, 1)
    b0_max = b0_aim + threshold 
    b1_min = max(b1_aim - threshold, 1)
    b1_max = b1_aim + threshold

    print("---------------------------------")
    print("Parameters for k = %s" % k)
    print("---------------------------------")

    # Loop through m, b0, b1
    for m in range(m_min, m_max + 1):
        for b0 in range(b0_min, b0_max + 1):
            for b1 in range(b1_min, b1_max + 1):
                # print("Testing m = %s, b0 = %s, b1 = %s" % (m, b0, b1))
                # Invalid range
                if 2**k + 2 < 2*m*(b0*b1+b0+1)+(1+4*m):
                    continue
                
                # repeater = repeater_type(k=k, m=m)
                # rate = repeater.get_rate([b0, b1], 300e3, None)
                rate, n = get_optimal_rate(k, m, b0, b1, L, repeater_type)
                if(m == m_aim and b0 == b0_aim and b1 == b1_aim):
                    print("Optimal Rate (table parameters): %s" % rate)
                # Update optimal Ns
                if rate_opt is None or rate_opt < rate:
                    rate_opt = rate
                    b_opt = [b0, b1]
                    m_opt = m
                    n_opt = n

    print("Optimal Rate: %s" % rate_opt)
    print("Optimal m: %s" % m_opt)
    print("Optimal b0: %s" % b_opt[0])
    print("Optimal b1: %s" % b_opt[1])
    print("Optimal n: %s" % n_opt)

    return rate_opt, b_opt, m_opt

def get_optimal_key_rate(k, L, repeaterType):

    if repeaterType == ImprovedRepeater:
        opt_params_dict = {
            7: (4, [4,2]),
            8: (5, [5,3]),
            9: (6, [7,4]),
            10: (8, [10,5])
        }
    else:
        # opt_params_dict = {
        #     7: (5, [3,2]),
        #     8: (8, [4,2]),
        #     9: (11, [5,3]),
        #     10: (12, [7,4])
        # }
        opt_params_dict = {
            7: (4, [4,2]),
            8: (5, [5,3]),
            9: (7, [6,4]),
            10: (8, [7,4])
        }
    optimal_rate = get_optimal_rate(k, opt_params_dict[k][0], opt_params_dict[k][1][0], opt_params_dict[k][1][1], L, repeaterType)[0]
    return optimal_rate

if __name__ == "__main__":
    # make_plot_1()
    # # get_optimal_params(7, ImprovedRepeater, 4, 4, 2)
    # # get_optimal_params(8, ImprovedRepeater, 5, 5, 3)
    # # get_optimal_params(9, ImprovedRepeater, 6, 7, 4)
    # # get_optimal_params(10, ImprovedRepeater, 8, 10, 5)

    # # print(get_optimal_rate(8, 4, 7, 3, 200e3, ImprovedRepeater))
    # # print(get_optimal_rate(8, 4, 7, 3, 400e3, ImprovedRepeater))
    # # print(get_optimal_rate(8, 4, 7, 3, 600e3, ImprovedRepeater))
    # # print(get_optimal_rate(8, 4, 7, 3, 800e3, ImprovedRepeater))

    params_7 = get_optimal_params(7, NaiveRepeater, 5, 3, 2)
    params_8 = get_optimal_params(8, NaiveRepeater, 8, 4, 2)
    params_9 = get_optimal_params(9, NaiveRepeater, 11, 5, 3)
    params_10 = get_optimal_params(10, NaiveRepeater, 12, 7, 4)

    # L_list = np.linspace(0, 500e3, 20)
    # key_rate_list_7 = [get_optimal_key_rate(7, L, NaiveRepeater) for L in L_list]
    # key_rate_list_8 = [get_optimal_key_rate(8, L, NaiveRepeater) for L in L_list]
    # key_rate_list_9 = [get_optimal_key_rate(9, L, NaiveRepeater) for L in L_list]
    # key_rate_list_10 = [get_optimal_key_rate(10, L, NaiveRepeater) for L in L_list]

    # plt.plot(L_list, key_rate_list_7, label="k = 7", color="gold")
    # plt.plot(L_list, key_rate_list_8, label="k = 8", color="purple")
    # plt.plot(L_list, key_rate_list_9, label="k = 9", color="green")
    # plt.plot(L_list, key_rate_list_10, label="k = 10", color="skyblue")
    
    # # make label next to line
    # plt.legend()

    # # log scale for y axis
    # plt.yscale("log")

    # # more ticks
    # plt.yticks([10**i for i in range(-9, 1, 1)])
    # plt.ylim(1e-9, 1)
    # plt.xlim(0, 500e3)

    # # Grid lines
    # plt.grid(True, which="both", axis="both")
    # plt.show()
    # # Save figure
    # plt.savefig("secret_key_bits_per_mode.png")
     