#lieu pour mettre les class ou fonction de test:
import numpy as np

from pricing import *
from quant import *

class Main:
    def __init__(self, param):
        pass

    @staticmethod
    def test_Hull_White():
        donne_simule = hullWhite(True, list(np.random.normal(0, 1, 1000)))
        t_liste = np.linspace(0, 1, 1000)
        print(len(donne_simule))
        print(len(t_liste))
        plotSimulation(t_liste, donne_simule)

    @staticmethod
    def HullWhite2(isForSimulation = False, dW = None, tau = 0.01, tn = 1):
        k = hullWhite(isForSimulation, dW, tau, tn)
        print(k)

    @staticmethod
    def testTrajectoire(n_traject=4, n_obser=100,T=1):
        trajectoires = generateurTrajectoire(n_traject, n_obser, T)
        print(trajectoires.head(2))
        plotSimulation(trajectoires.columns, list(trajectoires.T.values))

    @staticmethod
    def testSimulationTaux(N=4, T=1):
        simulation_Taux = simulationProcessusTaux(N, T, True)
        print(simulation_Taux)
        plotSimulation(simulation_Taux.columns, list(simulation_Taux.T.values))

    def testSimulationP(n, T=1):
        simulation_P = simulationP(n,T)
        print(simulation_P)
        simulation_P = pd.DataFrame(simulation_P)
        plotSimulation(simulation_P.columns, list(simulation_P.T.values))
    def testSimulationVrec(n,T=1):
        simulation_Vrec = simulationVrec(n,N=100,T=1,𝜏= 0.5)
        print(simulation_Vrec)
        simulation_Vrec = pd.DataFrame(simulation_Vrec)
        plotSimulation(simulation_Vrec.columns, list(simulation_Vrec.T.values))

    def courbe_fwdinst(t, α=0.1, sigma=0.15):
        x = np.linspace(1,t,100)
        print(x)
        courbe = []
        for j in x:
            courbe.append(MarketZeroCoupon.getmarketZeroCouponInstFwdCurve(j))
        print(courbe)
        plt.plot(x,courbe)
        plt.show()

#Main.courbe_fwdinst(10)
#Main.test_Hull_White()
Main.testTrajectoire()
#Main.testSimulationTaux(100)
#Main.testSimulationP(100)
#Main.testSimulationVrec(10)
