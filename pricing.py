#developpement de tout ce qui est en lien avec la partie pricing
#importation library
import numpy as np

from quant import *

#fonction qui prend en param le notionnel N, la date de valo, le taux fixe du swap
def pricingSwap():
    pass


#simulation de trajectoire
def generateurTrajectoire(n_traject, n_obser, T):
    dt = T / n_obser
    Normal_Matrix = np.random.normal(0, np.sqrt(dt), (n_traject, n_obser - 1))
    Brownien_process = Normal_Matrix.cumsum(axis=1)
    Brownien_process = np.insert(Brownien_process, 0, 0, axis=1)
    Brownien_process_Df = pd.DataFrame(data=Brownien_process, columns=np.linspace(0, T, n_obser), index=["trajectoire_" + str(i) for i in range(n_traject)])
    return Brownien_process_Df


#Simulation du processus
def simulationProcessusTaux(N=1000, T=1, isForSimulation = True):
    W = generateurTrajectoire(N+2, T)
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
# n = le nombre de trajectoires

# Il faut différencier n le nombre de trajectoires et le nombre d'échéances
# à t=0, le modèle ne marche pas à cause de la fonction de Siegel

def simulationP(n,T,t=0,trajectoire=0,isForSimulation = True):
    r = simulationProcessusTaux(n, T)
    r = np.array(r)
    if isForSimulation:
        t = T/n
        A = [0]
        B = [0]
        for i in range(1,n):
            A.append(getA(i*t,T, α = 0.1, sigma = 0.15))
            B.append(BondPrice(i*t, T, α=0.1))


        B = np.array(B)
        A = np.array(A)
        return (A*np.exp(-1*B*r))
    else:
        return(getA(t,T, α = 0.1, sigma = 0.15)*np.exp(-BondPrice(t, T, α=0.1)*r[trajectoire][t]))


#Simulation du Vrec
#N: notional
def simulationVrec(n,N,T,𝜏= 0.5):
    t = T/n
    P = simulationP(n,T)
    K = [0]*n
    L = P.copy()
    for i in range(n):
        S = sum([simulationP(n, T=j * t, t=1, trajectoire=i, isForSimulation=False) for j in range(1, n)])
        K[i] = (simulationP(n,T=1,t=1,trajectoire=i,isForSimulation = False) - (simulationP(n,T=n,t=1,trajectoire=i,isForSimulation = False)))/S
        for j in range(1,(n-1)):
            L[i][j] = (1 / 𝜏) * ((P[i][j] / P[i][j+1]) - 1)

    K = np.array(([K]*n))
    Vrec = N * 𝜏 * (K - L) * P
    return(Vrec)
