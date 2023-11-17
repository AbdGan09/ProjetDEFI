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
    Brownien_process_Df = pd.DataFrame(data=Brownien_process, columns=["t_" + str(i) for i in range(N)], index=["trajectoire_" + str(i) for i in range(N)])
    #dW_Brownien_process = Brownien_process_Df.diff(axis=1, periods=1) In case of
    #dW_Brownien_process["t_0"] = np.zeros(N)
    return Brownien_process_Df


#Simulation du processus
def simulationProcessusTaux(N=1000, T=1, isForSimulation = True):
    W =  generateurTrajectoire(N+2, T)
    dW = W.diff(axis=1, periods=1)
    tau = T/N
    for trajectoire in W.index:
        W.loc[trajectoire] = hullWhite(isForSimulation, list(dW.loc[trajectoire]), tau)
    rate_process = W[W.columns[1:N+1]]
    rate_process = rate_process.iloc[:N]
    rate_process.columns =["t_"+str(i) for i in range(N)]
    rate_process.index = ["trajectoire_" + str(i) for i in range(1,N+1)]
    return rate_process


#Simulation du discount
def simulationP():
    pass


#Simulation du Vrec
def simulationVrect():
    pass