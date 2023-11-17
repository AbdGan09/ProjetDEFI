#lieu pour mettre les class ou fonction de test:
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
    def testTrajectoire(N=1000, T=1):
        trajectoires = generateurTrajectoire(N, T)
        print(trajectoires.head(2))
        plotSimulation(trajectoires.columns, list(trajectoires.T.values))

    @staticmethod
    def testSimulationTaux(N=4, T=1):
        simulation_Taux = simulationProcessusTaux(N, T, True)
        print(simulation_Taux)
        plotSimulation(simulation_Taux.columns, list(simulation_Taux.T.values))

Main.test_Hull_White()
Main.testTrajectoire(100)
Main.testSimulationTaux(100)
