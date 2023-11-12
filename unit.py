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
    def HullWhite2():
        k = hullWhite(isForSimulation = False, dW = None, tau = 0.01, tn = 1, Sigma = 0.15, a = 0.1, seed = 42)
        print(k)
Main.test_Hull_White()
Main.HullWhite2()
