#developpement de tout ce qui est en lien avec la partie pricing
#importation library
import numpy as np
import math

from quant import *

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


#Simulation du discount
# à t=0, le modèle ne marche pas à cause de la fonction de Siegel
# P le prix actualisé par rapport au t_obs où on se positionne
# L le taux actualisé
# r la courbe des taux (si simulation=False elle est passé en paramètre)
# R le taux fixe
# i_traj correspond au numéro de la trajectoire voulu
def simulationP(n_traject, n_obser, T, r,R, t, 𝜏, i_traj=0, isPsimulation=False):
    if isPsimulation:
        r = simulationProcessusTaux(n_traject, n_obser, T)
        r = r.to_dict('index')
    else:
        dates_paiement = np.round(np.arange(0, (T / 𝜏)+1, 𝜏), 3)
        P = []
        L = []

        p_ = {}
        l_ = {}

        j0=dates_paiement[0]
        for j in dates_paiement[1:len(dates_paiement)-1]:
            if t<= j:
                t = float(t)

                p = zeroCoupon(t, j, r['trajectoire_'+str(i_traj)][t])
                l = R - (1 / 𝜏) * ((zeroCoupon(t, j, r['trajectoire_'+str(i_traj)][t]) /zeroCoupon(t, j, r['trajectoire_' + str(i_traj)][round(t-round(t-j0,1),1)])) - 1)

                j0 =t
                p_[j] = p
                l_[j] = l
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
#K: les différents t observé selon le nombre d'observations choisi
#i_traj: correcpond au numéro de la trajectoire voulu
#v_t: la somme du produit du prix actualisé par le taux actualisé
def simulationVrec(n_traject, n_obser, N, T, R=0.03, 𝜏=0.5):
    r = simulationProcessusTaux(n_traject, n_obser, T)
    r = r.to_dict('index')
    Vrec = {}
    K = r['trajectoire_0'].keys()

    for i_traj in range(n_traject):
        V = {}
        for t_obs in K:
            v_t = 0
            L, P= simulationP(n_traject, n_obser, T,r, R, t_obs, 𝜏, i_traj)
            prix_actual = np.array(P.iloc[0]).T
            L_actual = np.array(L.iloc[0])
            v_t = np.dot(L_actual,prix_actual)
            V[t_obs] = v_t * N

        Vrec[i_traj] = V
    Vrec = pd.DataFrame(Vrec).T
    return Vrec


