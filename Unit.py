from quant import *
from Pricing import *
class main :
    def __init__ (self, nbSimulation):
        self.nbSimulation = nbSimulation
        self.simulation =simulationWiener(self.nbSimulation)
        self.simulation
print(main(100).simulation)