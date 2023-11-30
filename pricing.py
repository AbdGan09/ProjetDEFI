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
    rate_process.columns =[ round(i*dt,3) for i in range(n_obser)]
    rate_process.index = ["trajectoire_" + str(i) for i in range(0, n_traject)]
    return rate_process

from decimal import localcontext, Decimal, ROUND_HALF_UP
# Définir le contexte décimal avec l'arrondi vers le haut (ROUND_HALF_UP)
#localcontext().rounding = ROUND_HALF_UP
def custom_round(value, decimals=3):
    return Decimal(str(value)).quantize(Decimal('1e-{0}'.format(decimals)),rounding = ROUND_HALF_UP)

#Simulation du discount
# à t=0, le modèle ne marche pas à cause de la fonction de Siegel
# t une date de paiement antérieure à T la maturité
def simulationP(n_traject,n_obser, T, R, t, 𝜏):
    r = simulationProcessusTaux(n_traject, n_obser, T)
    iterations = r.columns
    r = r.to_dict('index')
    dt = float(custom_round(custom_round(custom_round(T / n_obser),3)))
    iterations += dt
    iterations = np.round(iterations,3)
    print(iterations)
    iteration = [0.0] + iterations.tolist()
    v=t
    P=[]
    L=[]
    for m in range(n_traject):
        print('m',m)
        p_ = {}
        l_ = {}
        j=dt
        w=j
        for j in iterations:
            t=float(v)
            i = list(iteration).index(t)
            p = []
            l = []
            for t in iteration[i:]:
                w=j
                if j <= t:
                    p.append(getA(j,t, α = 0.1, sigma = 0.15)*math.exp(-1*BondPrice(j, t, α=0.1)*r['trajectoire_'+str(m)][round(w-dt,3)]))
                    l.append(R-(1/𝜏)*((getA(j,t-𝜏, α = 0.1, sigma = 0.15)*math.exp(-1*BondPrice(j, t-𝜏, α=0.1)*r['trajectoire_'+str(m)][round(w-dt,3)])/getA(j,t, α = 0.1, sigma = 0.15)*math.exp(-1*BondPrice(j, t, α=0.1)*r['trajectoire_'+str(m)][round(w-dt,3)]))-1))
            p_[j]=p
            l_[j]=l
        P.append(p_)
        L.append(l_)
        m += 1
    K = P
    P = pd.DataFrame(P)
    L = pd.DataFrame(L)

    return (L, P, K)

#Simulation du Vrec
#N: notional
def simulationVrec(n_traject,n_obser, N, T, r=0.03,𝜏= 0.5):
    dt = float(custom_round(custom_round(custom_round(T / n_obser),3)))
    Vrec = {}
    L, P, K = simulationP(n_traject, n_obser, T, r, 1, 𝜏)
    for m in range(n_traject):
        V = {}
        for t_obs in K[0].keys():
            d=0
            for i in  [0.0]+[k+𝜏 for k in range(T)]+[float(T)]:
                #print('i',i)
                L,P,_ = simulationP(n_traject,n_obser, T,r, i, 𝜏)

                if L.iloc[m][t_obs]!=[] and P.iloc[m][t_obs]!=[]:
                    g = P.iloc[m][t_obs]
                    h = L.iloc[m][t_obs]
                    d+=N*(h[0]*g[0])
            V[t_obs]=d
        #print('mVrec',m)
        Vrec[m]=V
    Vrec = pd.DataFrame(Vrec).T
    return(Vrec)

