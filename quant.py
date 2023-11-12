#importation des fichier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
import scipy.misc
import warnings
warnings.filterwarnings("ignore")

##methode à utiliser après###

#definition de la class que monsieur a demander dans son cours:
class MarketZeroCoupon:
    def __init__(self):
        pass

    @staticmethod
    def getMarketZeroCouponCurve(T, Betha0 = 2.61, Betha1 = -1.33, lambdas = 0.17):
        return np.exp(-((Betha0+Betha1*((1-np.exp(-lambdas*T))/(lambdas*T)))*(T/100)))

    @staticmethod
    def getmarketZeroCouponInstFwdCurve(t):
        f = lambda x: -np.log(MarketZeroCoupon.getMarketZeroCouponCurve(x))
        return scipy.misc.derivative(f, t, dx = 1.25)

#plot des trajectoire
def plotSimulation(Donne_Simule):
    Donne_Simule.plot()
    return 0

#fonction A(t,T)
def A(t,T, α = 0.1, sigma = 0.15):
    B = BondPrice(t, T)
    fM = MarketZeroCoupon.getmarketZeroCouponInstFwdCurve(t)
    e = np.exp((-B*fM)-((sigma**2*((np.exp(-α*T)-np.exp(-α*t))**2)*(np.exp(2*α*t)-1))/(4*α**3)))
    return ((MarketZeroCoupon.getMarketZeroCouponCurve(T)/MarketZeroCoupon.getMarketZeroCouponCurve(t))*e)


#fonction Theta
def gettheta(t, α = 0.1, sigma = 0.15):
    f = lambda x: MarketZeroCoupon.getmarketZeroCouponInstFwdCurve(x)
    dfM = scipy.misc.derivative(f, t, dx=1.25)
    fM = MarketZeroCoupon.getmarketZeroCouponInstFwdCurve(t)
    return ((α*fM)+dfM+(((sigma**2)/2*α)*(1-np.exp(-2*α*t))))


#fonction B(t, T)
def BondPrice(t, T, α = 0.1):
    return ((1 - np.exp(-α * (T - t))) / α)


#fonction P(t,T): The Zero Coupon price
#T: maturité
#t: pour le temps actuel
def zeroCoupon(t, T):
    return A(t, T) * np.exp(-BondPrice(t, T) * hullWhite(t))


#Modele de Hull White
# dW: liste de simulation de la loi normal centré réduite.
#IsForSimulation à True si vous voulez utilisez cette fonction dans la partie pricing
#le tau sera modifié une fois les fonction réajusté
# l'autre partie est juste pour le calcul ponctuelle sans utilisé de liste et en utilisant la récursivité
# le 2 sera ajusté également.
def hullWhite(isForSimulation = False, dW = None, tn = None, tau = 1, Sigma = 0.15, a = 0.1):
    if isForSimulation:
        rate = list(np.zeros(len(dW)))
        rate[0] = 0
        for i in range(2, len(dW)):
            rate[i] = rate[i-1] + Sigma * (gettheta((i-1)*tau) - a * rate[i-1]) + Sigma * tau * dW[i]
        return rate
    else:
        r0 = 0
        dW = np.random.normal()
        if tn == 0:
            return r0
        else:
            r_n_1 = hullWhite(tn-tau)
            return r_n_1 + Sigma * (gettheta(tn-tau) - a * r_n_1) + Sigma * tau * dW

print(hullWhite(True, list(np.random.normal(0, 1, 5))))