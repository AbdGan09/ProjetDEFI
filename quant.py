#importation des fichier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc

##methode à utiliser après###

#definition de la class que monsieur a demander dans son cours:
class MarketZeroCoupon:
    def __init__(self, parametre):
        self.parametre = parametre

    @staticmethod
    def getMarketZeroCouponCurve(parametre):
        return marketZeroCouponCurve(parametre)

    @staticmethod
    def getmarketZeroCouponInstFwdCurve(parametre):
        return marketZeroCouponInstFwdCurve(parametre)

#comme discuté j'ai juste écrire les classes les fonctions à l'intérieur on se les repartis


#si quelqu'un juge necessaire de faire des function suplémentaire il peux le faire l'objectif étant de mieu segmenté notre code pour le rendre plus lisible




#MarketZeroCouponCurve
def marketZeroCouponCurve(T, Betha0 = 2.61, Betha1 = -1.33, lambdas = 0.17):
    return np.exp(-((Betha0+Betha1*((1-np.exp(-lambdas*T))/(lambdas*T)))*(T/100)))



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
#T: maturité
#t: pour le temps actuel
def zeroCoupon(t, T):
    return A(t, T) * np.exp(-B(t,T)*hullWhite(t))




#Modele de Hull White
def hullWhite():
    pass
