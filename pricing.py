#developpement de tout ce qui est en lien avec la partie pricing
#importation library

from quant import *

#fonction qui prend en param le notionnel N, la date de valo, le taux fixe du swap
def pricingSwap():
    pass


#simulation de trajectoire
def generateurTrajectoire():
    pass


#Simulation du processus
def simulationProcessusTaux():
    pass


#Simulation du discount
def simulationP(t,T):
    A = A(t,T, α = 0.1, sigma = 0.15)
    B = BondPrice(t, T, α=0.1)
    r = simulationProcessusTaux(t,T)
    return (A*np.exp(-B*r))


#Simulation du Vrec
def simulationVrect(N,K,t,T,𝜏= 0.5):
    Vrec = []
    for i in range(1,T+1):
        if i>t:
            L = (1/𝜏)*((simulationP(t,i-1)/simulationP(t,i))-1)
            Vrec+=N*𝜏*(K-L)*simulationP(t,i)
    return(Vrec)
