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

def simulationP(n_traject, n_obser, T, R, t, 𝜏, m=None):
    r = simulationProcessusTaux(n_traject, n_obser, T)
    r = r.to_dict('index')
    iterations = np.round(np.arange(0, T + T / n_obser, T / n_obser), 3)
    P = []
    L = []

    if m is None:
        for m in range(n_traject):
            p_ = {}
            l_ = {}
            j0 = iterations[0]
            for j in iterations[1:len(iterations) - 1]:
                t = float(t)

                p = zeroCoupon(j, T, r['trajectoire_' + str(m)][j])
                l = R - (1 / 𝜏) * ((zeroCoupon(j, T, r['trajectoire_' + str(m)][j]) / zeroCoupon(j, T, r['trajectoire_' + str(m)][round(j - round(j - j0, 1), 1)])) - 1)

                j0 = j
                p_[j] = p.tolist()
                l_[j] = l.tolist()

            P.append(p_)
            L.append(l_)

    else:
        p_ = {}
        l_ = {}

        j0=iterations[0]
        for j in iterations[1:len(iterations)-1]:
            t = float(t)

            p = zeroCoupon(j, T, r['trajectoire_'+str(m)][j])
            l = R - (1 / 𝜏) * ((zeroCoupon(j, T, r['trajectoire_'+str(m)][j]) /zeroCoupon(j, T, r['trajectoire_' + str(m)][round(j-round(j-j0,1),1)])) - 1)

            j0 =j
            p_[j] = p.tolist()
            l_[j] = l.tolist()

        P.append(p_)
        L.append(l_)

    K = P
    P = pd.DataFrame(P)
    L = pd.DataFrame(L)

    return L, P, K

#Simulation du Vrec
#N: notional
def simulationVrec(n_traject, n_obser, N, T, r=0.03, 𝜏=0.5):
    iterations = np.round(np.arange(0, T + T / n_obser, T / n_obser), 3)
    Vrec = {}

    L, P, K = simulationP(n_traject, n_obser, T, r, 1, 𝜏, 0)

    for m in range(n_traject):
        V = {}
        for t_obs in K[0].keys():
            d = 0
            for i in np.concatenate(([0.0], np.arange(𝜏, T, 𝜏))):
                if t_obs<= i:
                    L, P, _ = simulationP(n_traject, n_obser, T, r, i, 𝜏, m)
                    g = P.iloc[0][t_obs]
                    h = L.iloc[0][t_obs]
                    d += N * (h * g)
            V[t_obs] = d

        Vrec[m] = V
    Vrec = pd.DataFrame(Vrec).T
    return Vrec


