#developpement de tout ce qui est en lien avec la partie pricing
#importation library

from quant import *

#fonction qui prend en param le notionnel N, la date de valo, le taux fixe du swap
def pricingSwap():
    pass


#simulation de trajectoire
def generateurTrajectoire(N, T):
    dt = T / N
    Normal_Matrix = np.random.normal(0, np.sqrt(dt), (N, N - 1))
    Brownien_process = Normal_Matrix.cumsum(axis=1)
    Brownien_process = np.insert(Brownien_process, 0, 0, axis=1)
    Brownien_process_Df = pd.DataFrame(data=Brownien_process, columns=["t_" + str(i) for i in range(N)],
                                       index=["trajectoire_" + str(i) for i in range(N)])
    #dW_Brownien_process = Brownien_process_Df.diff(axis=1, periods=1) In case of
    #dW_Brownien_process["t_0"] = np.zeros(N)
    return Brownien_process_Df


#Simulation du processus
def simulationProcessusTaux():
    pass


#Simulation du discount
def simulationP():
    pass


#Simulation du Vrec
def simulationVrect():
    pass