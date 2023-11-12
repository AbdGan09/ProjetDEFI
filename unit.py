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

Main.test_Hull_White()

