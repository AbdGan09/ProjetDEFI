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

def simulationP(n_traject, n_obser, T, r,R, t, 𝜏, m=0, simulation=False):
    if simulation:
        r = simulationProcessusTaux(n_traject, n_obser, T)
        r = r.to_dict('index')
    else:
        iterations = np.round(np.arange(0, (T / 𝜏)+1, 𝜏), 3)
        P = []
        L = []

        p_ = {}
        l_ = {}

        j0=iterations[0]
        for j in iterations[1:len(iterations)-1]:
            if t<= j:
                t = float(t)

                p = zeroCoupon(t, j, r['trajectoire_'+str(m)][t])
                l = R - (1 / 𝜏) * ((zeroCoupon(t, j, r['trajectoire_'+str(m)][t]) /zeroCoupon(t, j, r['trajectoire_' + str(m)][round(t-round(t-j0,1),1)])) - 1)

                j0 =t
                p_[j] = p.tolist()
                l_[j] = l.tolist()
            else:
                p_[j] = 0.0
                l_[j] = 0.0

        P.append(p_)
        L.append(l_)



    P = pd.DataFrame(P)
    L = pd.DataFrame(L)

    return L, P

#Simulation du Vrec
#N: notional
def simulationVrec(n_traject, n_obser, N, T, R=0.03, 𝜏=0.5):
    r = simulationProcessusTaux(n_traject, n_obser, T)
    r = r.to_dict('index')
    Vrec = {}
    K = r['trajectoire_0'].keys()

    for m in range(n_traject):
        V = {}
        for t_obs in K:
            d = 0
            L, P= simulationP(n_traject, n_obser, T,r, R, t_obs, 𝜏, m)
            g = np.array(P.iloc[0]).T
            h = np.array(L.iloc[0])
            d = np.dot(h,g)
            V[t_obs] = d * N

        Vrec[m] = V
    Vrec = pd.DataFrame(Vrec).T
    return Vrec


