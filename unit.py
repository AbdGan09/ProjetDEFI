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
    def testTrajectoire(n_traject=4,n_obser=1000,T=30):
        n_obser = n_obser*T
        trajectoires = generateurTrajectoire(n_traject, n_obser,T, 𝜏= 0.5)
        print(trajectoires.head(2))
        plotSimulation(trajectoires.columns, list(trajectoires.T.values))

    @staticmethod
    def testSimulationTaux(n_traject=4, n_obser=100, T=10):
        n_obser = T*n_obser
        simulation_Taux = simulationProcessusTaux(n_traject, n_obser, T, True)
        print(simulation_Taux)
        plotSimulation(simulation_Taux.columns, list(simulation_Taux.T.values))

    def testSimulationP(n_traject=1, n_obser=100, T=10,R=0.03, t=4.5, 𝜏=0.5):
        #n_obser = T * n_obser
        L,simulation_P,_ = simulationP(n_traject,n_obser, T,R, t, 𝜏)
        #simulation_P.to_clipboard()
        #L.to_clipboard()
        print(simulation_P)
        print(L)
        #simulation_P = pd.DataFrame(simulation_P)
        #plotSimulation(simulation_P.columns, list(simulation_P.T.values))
    def testSimulationVrec(n_traject=3,n_obser=300, N=100, T=30, r=0.03,𝜏= 1):
        simulation_Vrec = simulationVrec(n_traject,n_obser, N, T, r=0.03,𝜏= 1)
        print(simulation_Vrec.mean(axis=0))
        print(simulation_Vrec)
        simulation_Vrec.T.plot()
        simulation_Vrec.mean(axis=0).T.plot()
        plt.show()
        #simulation_Vrec = pd.DataFrame(simulation_Vrec)
        #plotSimulation(simulation_Vrec.columns, list(simulation_Vrec.values))

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
#Main.testTrajectoire()
#Main.testSimulationTaux()
#Main.testSimulationP()
Main.testSimulationVrec()
