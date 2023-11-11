#importation des fichier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
import scipy.misc

##methode à utiliser après###

#definition de la class que monsieur a demander dans son cours:
class MarketZeroCoupon:
    def __init__(self):
        pass

    @staticmethod
    def getMarketZeroCouponCurve(T, Betha0 = 2.61, Betha1 = -1.33, lambdas = 0.17):
        return np.exp(-((Betha0+Betha1*((1-np.exp(-lambdas*T))/(lambdas*T)))*(T/100)))

    @staticmethod
    def getmarketZeroCouponInstFwdCurve(T):
        f = lambda x: -np.log(MarketZeroCoupon.getMarketZeroCouponCurve(T))
        return scipy.misc.derivative(f, T)

#comme discuté j'ai juste écrire les classes les fonctions à l'intérieur on se les repartis


#si quelqu'un juge necessaire de faire des function suplémentaire il peux le faire l'objectif étant de mieu segmenté notre code pour le rendre plus lisible


#plot des trajectoire
def plotSimulation():
    pass


#fonction A(t,T)
def A():
    pass


#fonction Theta
def gettheta():
    return Theta


#fonction B(t, T)
def BondPrice(t, T, α=0.1):
    return ((1 - np.exp(-α * (T - t))) / α)


#fonction P(t,T): The Zero Coupon price
#T: maturité
#t: pour le temps actuel
def zeroCoupon(t, T):
    return A(t, T) * np.exp(-BondPrice(t,T)*hullWhite(t))




#Modele de Hull White
def hullWhite():
    pass
