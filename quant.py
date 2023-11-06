#importation des fichier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc

##methode à utiliser après###

#definition de la class que monsieur a demander dans son cours:
class MarketZeroCoupon:
    def __init__(self, x):
        self.x = x

    def MarketZeroCouponCurve(self, t):
        if t == 0:
            return 1
        else:
            numerator = 1 - math.exp(-self.x * t)
            denominator = self.x * t
            return 1 + (numerator / denominator)

    def MarketZeroCouponInstFwdCurve(self, t):
        if t == 0:
            return 0
        else:
            P_t = self.MarketZeroCouponCurve(t)
            return -P_t / (self.x * (t ** 2))

    def print_curve_values(self, t):
        P_t = self.MarketZeroCouponCurve(t)
        fM_t = self.MarketZeroCouponInstFwdCurve(t)
        print(f"P({t}) = {P_t}")
        print(f"fM({t}) = {fM_t}")
    

#comme discuté j'ai juste écrire les classes les fonctions à l'intérieur on se les repartis


#si quelqu'un juge necessaire de faire des function suplémentaire il peux le faire l'objectif étant de mieu segmenté notre code pour le rendre plus lisible




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
