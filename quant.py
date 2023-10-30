#importation des packages
import numpy as np
import pandas as pd

# Coder la Class MarketZeroCoupon

class marketZeroCoupon:
    def __init__(self, param):
        self.param = param


    @staticmethod
    def marketZeroCouponCurve(param):
        return 1 + (1+np.exp(-param))/param

    @staticmethod
    def marketZeroCouponInstFwdCurve(param):
        return np.exp(-param)*(param+1)/(param*marketZeroCoupon(param))


def simulationWiener(nbSimulation, T = 1, N = 200):
    dt = T/N
    dataSimule = {}
    for i in range(nbSimulation):
        scenario = np.random.normal(0, np.sqrt(dt), N+1)
        dataSimule[str(i)] = scenario
    dataSimule = pd.DataFrame(dataSimule)
    firstLine = pd.Series(list(np.zeros(nbSimulation)))
    dataSimule.iloc[0] = firstLine
    dataSimule = dataSimule.T
    dataSimule.columns = [i*dt for i in range(N+1)]
    return dataSimule


def simulationProcessusTaux(nbSimulation, N, sigma, a, alpha):
    dt = 1/N
    processuss = simulationWiener(nbSimulation, N)
    for i in range(nbSimulation):
        for j in range(1, N):
             theta = getTheta(j*dt, sigma, a)
             processuss.iloc[i, j] = processuss.iloc[i, j-1] + (theta - alpha * processuss.iloc[i, j-1])*dt + sigma * processuss.iloc[i, j]

def simulationP(nbSimulation, N, sigma, a, alpha):
    processuss = simulationProcessusTaux(nbSimulation, N, sigma, a, alpha)
    for i in range(nbSimulation):
        for j in range(1, N):
            processuss.iloc[i, j] = 0

def plotSimulation(nbSimulation, N):
    return simulationWiener(nbSimulation, N).T.plot()

def getTheta(param, sigma, a):
    firsPart = a * marketZeroCoupon.marketZeroCouponInstFwdCurve(param)
    secondPart = getDerive(param, marketZeroCoupon.marketZeroCouponCurve(param))
    thirdPart = (sigma**2)/(2*a)*(1-np.exp(-2*a*param))
    return firsPart + secondPart + thirdPart


# import package
import numpy as np


def getDate(date):
    return date


def getDerive(param, fonction):
    return (param ** 2 * fonction * np.exp(-param) - 2 * fonction * (param + 1) * np.exp(-param) + np.exp(
        -2 * param) * (param + 1) ** 2 / param ** 2) / (param ** 4 * fonction ** 2)
