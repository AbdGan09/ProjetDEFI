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
    def __init__(self, parametre):
        self.parametre = parametre

    @staticmethod
    def getMarketZeroCouponCurve(parametre):
        return marketZeroCouponCurve(parametre)

    @staticmethod
    def getmarketZeroCouponInstFwdCurve(T, Betha0 = 2.61, Betha1 = -1.33, lambdas = 0.17):
        f = lambda x: -np.log(getMarketZeroCouponCurve(x, Betha0 = 2.61, Betha1 = -1.33, lambdas = 0.17))
        return scipy.misc.derivative(f, T)

#MarketZeroCouponCurve
def marketZeroCouponCurve():
    pass


def marketZeroCouponInstFwdCurve():
    pass


#definition de la derivee Seconde a etre utilise après pour le calcul de Theta
def getSecondDerive(param, fonction):
    pass


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
def B():
    pass


#fonction P(t,T): The Zero Coupon price
def zeroCoupon():
    pass


#Modele de Hull White
def hullWhite():
    pass
