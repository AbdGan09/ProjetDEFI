#developpement de tout ce qui est en lien avec la partie pricing
#importation library
import numpy as np
import math

from quant import *

#fonction qui prend en param le notionnel N, la date de valo, le taux fixe du swap
def pricingSwap():
    pass


#simulation de trajectoire
def generateurTrajectoire(n_traject, n_obser,T, 𝜏= 0.5):
    dt = T / n_obser
    Normal_Matrix = np.random.normal(0, np.sqrt(dt), (n_traject, n_obser - 1))
    print(Normal_Matrix)
    Brownien_process = Normal_Matrix.cumsum(axis=1)
    Brownien_process = np.insert(Brownien_process, 0, 0, axis=1)
    Brownien_process_Df = pd.DataFrame(data=Brownien_process, columns=np.linspace(0, T, n_obser), index=["trajectoire_" + str(i) for i in range(n_traject)])
    return Brownien_process_Df


#Simulation du processus
def simulationProcessusTaux(n_traject, n_obser, T=1,isForSimulation = True):
    W = generateurTrajectoire(n_traject, n_obser+2, T)
    dW = W.diff(axis=1, periods=1)
    dt = T/n_obser
    for trajectoire in W.index:
        W.loc[trajectoire] = hullWhite(isForSimulation, list(dW.loc[trajectoire]), dt)
    rate_process = W[W.columns[1:n_obser+1]]
    rate_process = rate_process.iloc[:n_traject]
    rate_process.columns =[round(i*dt,1) for i in range(n_obser)]
    rate_process.index = ["trajectoire_" + str(i) for i in range(0, n_traject)]
    return rate_process


#Simulation du discount
# à t=0, le modèle ne marche pas à cause de la fonction de Siegel
# t une date de paiement antérieure à T la maturité
def simulationP(n_traject,n_obser, T, R, t, 𝜏):
    r = simulationProcessusTaux(n_traject, n_obser, T)
    r = r.to_dict('index')
    dt = T / n_obser
    v=t
    P=[]
    L=[]
    p_={}
    l_={}
    for m in range(n_traject):
        j=dt
        w=round(j,1)
        while j <= T:
            t=v
            p = []
            l = []
            while t<=T :
                if j <= t:
                    w = round(j, 1)
                    p.append(getA(j,t, α = 0.1, sigma = 0.15)*math.exp(-1*BondPrice(j, t, α=0.1)*r['trajectoire_'+str(m)][w-dt]))
                    l.append(R-(1/𝜏)*((getA(j,t-𝜏, α = 0.1, sigma = 0.15)*math.exp(-1*BondPrice(j, t-𝜏, α=0.1)*r['trajectoire_'+str(m)][w-dt])/getA(j,t, α = 0.1, sigma = 0.15)*math.exp(-1*BondPrice(j, t, α=0.1)*r['trajectoire_'+str(m)][w-dt]))-1))
                    t+=𝜏
                else:
                    t+=𝜏
            p_[j]=p
            l_[j]=l
            j += dt
        P.append(p_)
        L.append(l_)
    K = P
    P = pd.DataFrame(P)
    L = pd.DataFrame(L)
    return (L, P, K)

def multiply_arrays(array1, array2):
    result = [(a * b if isinstance(a, list) and isinstance(b, list) else []) if a and b else [] for a, b in zip(array1, array2)]
    return result


#Simulation du Vrec
#N: notional
def simulationVrec(n_traject,n_obser, N, T, r=0.03,𝜏= 0.5):
    dt = T / n_obser
    Vrec = {}
    L, P, K = simulationP(n_traject, n_obser, T, r, 1, 𝜏)
    for m in range(n_traject):
        V = {}
        #print(V)
        for t_obs in K[0].keys():
            d=0
            for i in range(int(T/𝜏)-1):
                L,P,_ = simulationP(n_traject,n_obser, T,r, i, 𝜏)

                #print(L.iloc[m][t_obs])
                #print('somme',sum(sum(np.multiply(np.array([L.iloc[m][t_obs]]),np.array([P.iloc[m][t_obs]])))))
                #V[t_obs]=N*sum(sum(np.multiply(np.array([L.iloc[m][t_obs]]),np.array([P.iloc[m][t_obs]]))))
                if L.iloc[m][t_obs]!=[] and P.iloc[m][t_obs]!=[]:
                    g = P.iloc[m][t_obs]
                    h = L.iloc[m][t_obs]
                    d+=N*(h[0]*g[0])
            V[t_obs]=d
            #print(V)

        Vrec[m]=V
        #Vrec.append(F)
    Vrec = pd.DataFrame(Vrec).T
    #print(Vrec)
    return(Vrec)

