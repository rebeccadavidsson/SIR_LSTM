import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class SIR():

    def __init__(self, N, I0, R0, beta, gamma, days):
        self.N      =  N # Total population, N.
        self.I0     = I0  # Initial number of infected 
        self.R0     = R0  # Initial number of recovered individuals
        self.S0     = N - I0 - R0 # Everyone else, S0, is susceptible to infection initially.
        self.beta   = beta # Contact rate, beta  (in 1/days)
        self.gamma  = gamma # Mean recovery rate, gamma  (in 1/days)
        self.days   = days
    
    def simulate(self, target="Infected", plot=True):
        N, I0, R0, S0, beta, gamma = self.N, self.I0, self.R0, self.S0, self.beta, self.gamma
        # print(N, I0, R0, S0, beta, gamma)

        # A grid of time points (in days)
        t = np.linspace(0, self.days, self.days)

        # The SIR model differential equations.
        def deriv(y, t, N, beta, gamma):
            S, I, R = y
            dSdt = -beta * S * I / N
            dIdt = beta * S * I / N - gamma * I
            dRdt = gamma * I
            return dSdt, dIdt, dRdt

        # Initial conditions vector
        y0 = S0, I0, R0
        # Integrate the SIR equations over the time grid, t.
        ret = odeint(deriv, y0, t, args=(N, beta, gamma))
        S, I, R = ret.T
        # Plot the data on three separate curves for S(t), I(t) and R(t)
        FACTOR = 1
            
        if plot:
            fig = plt.figure(facecolor='w')
            ax = fig.add_subplot(111, axisbelow=True)
            if target == "Infected" or target == "I":
                ax.plot(t, I/FACTOR, 'r', alpha=0.5, lw=2, label='Infected')
            elif target == "Susceptible" or target == "S":
                ax.plot(t, S/FACTOR, 'b', alpha=0.5, lw=2, label='Susceptible')
            elif target == "Recovered" or target == "R":
                ax.plot(t, R/FACTOR, 'g', alpha=0.5, lw=2, label='Recovered')
            else:
                ax.plot(t, I/FACTOR, 'r', alpha=0.5, lw=2, label='Infected')
                # ax.plot(t, S/FACTOR, 'b', alpha=0.5, lw=2, label='Susceptible')
                ax.plot(t, R/FACTOR, 'g', alpha=0.5, lw=2, label='Recovered')
            ax.set_xlabel('Time /days')
            ax.set_ylabel('Individuals')
            ax.yaxis.set_tick_params(length=0)
            ax.xaxis.set_tick_params(length=0)
            ax.grid(b=True, which='major', c='w', lw=2, ls='-')
            legend = ax.legend()
            legend.get_frame().set_alpha(0.5)
            for spine in ('top', 'right', 'bottom', 'left'):
                ax.spines[spine].set_visible(False)
            plt.show()
        return {"I": I, "S": S, "R": R}