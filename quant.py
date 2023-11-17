#importation des fichier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
import scipy.misc
import warnings
warnings.filterwarnings("ignore")


class MarketZeroCoupon:
    def __init__(self):
        pass

    @staticmethod
    def getMarketZeroCouponCurve(T, Betha0 = 2.61, Betha1 = -1.33, lambdas = 0.17):
        return np.exp(-((Betha0+Betha1*((1-np.exp(-lambdas*T))/(lambdas*T)))*(T/100)))

    @staticmethod
    def getmarketZeroCouponInstFwdCurve(t):
        f = lambda x: -np.log(MarketZeroCoupon.getMarketZeroCouponCurve(x))
        return scipy.misc.derivative(f, t, dx = 1e-3)


def plotSimulation(t_liste, Donne_Simule):
    plt.plot(t_liste, Donne_Simule)
    plt.title('Simulation')
    plt.xlabel('Temps')
    plt.ylabel('Données Simulées')
    plt.legend()
    plt.show()


# fonction A(t,T)
def A(t,T, α = 0.1, sigma = 0.15):
    B = BondPrice(t, T)
    fM = MarketZeroCoupon.getmarketZeroCouponInstFwdCurve(t)
    e = np.exp((-B*fM)-((sigma**2*((np.exp(-α*T)-np.exp(-α*t))**2)*(np.exp(2*α*t)-1))/(4*α**3)))
    return ((MarketZeroCoupon.getMarketZeroCouponCurve(T)/MarketZeroCoupon.getMarketZeroCouponCurve(t))*e)


# fonction Theta
def gettheta(t, α = 0.1, sigma = 0.15):
    f = lambda x: MarketZeroCoupon.getmarketZeroCouponInstFwdCurve(x)
    dfM = scipy.misc.derivative(f, t, dx = 1e-3)
    fM = MarketZeroCoupon.getmarketZeroCouponInstFwdCurve(t)
    return ((α*fM)+dfM+(((sigma**2)/2*α)*(1-np.exp(-2*α*t))))


# fonction B(t, T)
def BondPrice(t, T, α = 0.1):
    return ((1 - np.exp(-α * (T - t))) / α)


# fonction P(t,T): The Zero Coupon price
# T: maturité
# t: pour le temps actuel
def zeroCoupon(t, T):
    return A(t, T) * np.exp(-BondPrice(t, T) * hullWhite(t))


# Modele de Hull White
# dW: liste de simulation de la loi normal centré réduite.
# IsForSimulation à True si vous voulez utilisez cette fonction dans la partie pricing
# le tau sera modifié une fois les fonction réajusté
# l'autre partie est juste pour le calcul ponctuelle sans utilisé de liste et en utilisant la récursivité
# le 2 sera ajusté également.
def hullWhite(isForSimulation = False, dW = None, tau = 0.01, tn = None, Sigma = 0.15, a = 0.1, seed = 42):
    np.random.seed(seed)
    if isForSimulation:
        rate = list(np.zeros(len(dW)))
        for i in range(1, len(dW)-1):
            rate[i+1] = rate[i] + (gettheta((i)*tau) - a * rate[i]) * tau + Sigma * np.sqrt(tau) * dW[i]
        return rate
    else:
        r0 = 0
        dW = np.random.normal()
        if tn <= 0:
            return r0
        else:
            r_n_1 = hullWhite(isForSimulation = False, dW = dW, tau = 0.01, tn = tn-tau)
            return r_n_1 + (gettheta(tn-tau) - a * r_n_1) * tau + Sigma * np.sqrt(tau) * dW
