#developpement de tout ce qui est en lien avec la partie pricing
#importation library
import numpy as np
import math
from decimal import localcontext, Decimal, ROUND_HALF_UP
from quant import *

#fonction qui prend en param le notionnel N, la date de valo, le taux fixe du swap
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


def simUnitP(t, rate, T, 𝜏 = 0.5, Nb_payement = 60):
    P = []
    𝜏 = T/Nb_payement
    for i in range(1, Nb_payement+2):
        if t > i * 𝜏:
            P.append(None)
        else:
            P.append(zeroCoupon(t, i * 𝜏, rate))
    return P

#le init sera a suprimé après vue qu'il a dis de considéré le payement 1 directement et de  ne pas vraiment marqué le temps de strike.
#je me suis rendu compte qu'il y'a un petit tipo au niveau d'une multiplication donc la cloche est un peu décalé.
def simulationPV2(n_traject, n_obser, T, Nb_payement = 60):
    rate_ = list(simulationProcessusTaux(n_traject, n_obser, T).values)
    PV_sim = list(np.empty((n_traject, n_obser), dtype=object))
    dt = T/n_obser
    for index, traject_number in enumerate(rate_):
        for i, rate in enumerate(traject_number):
            PV_sim[index][i] = simUnitP(i*dt, rate, T)
    PV_sim = np.concatenate(PV_sim, axis = 0)
    Data = []
    for liste in PV_sim:
        Data.append(liste[:])
    Index1_trajectoir_liste = [["trajectoire_" + str(i)] * n_obser for i in range(n_traject)]
    Index1_trajectoir = [element for liste in Index1_trajectoir_liste for element in liste]
    Colonne = ["init"]+[("Payement_" + str(i)) for i in range(1, Nb_payement+1)]
    PV_sim_data_frame = pd.DataFrame(Data, columns=Colonne)
    PV_sim_data_frame["trajectoire"] = Index1_trajectoir
    PV_sim_data_frame["temps"] = [round(i*dt, 3) for i in range(n_obser)]*n_traject
    PV_sim_data_frame = PV_sim_data_frame.set_index(["trajectoire", "temps"])
    #PV_sim_data_frame.to_csv("check4.csv")
    return PV_sim_data_frame

# Définir le contexte décimal avec l'arrondi vers le haut (ROUND_HALF_UP)
#localcontext().rounding = ROUND_HALF_UP
def custom_round(value, decimals=3):
    return Decimal(str(value)).quantize(Decimal('1e-{0}'.format(decimals)),rounding = ROUND_HALF_UP)

#Simulation du discount
# à t=0, le modèle ne marche pas à cause de la fonction de Siegel
# t une date de paiement antérieure à T la maturité
def simulationP(n_traject,n_obser, T, R, t, 𝜏, m=None):
    r = simulationProcessusTaux(n_traject, n_obser, T)
    iterations = r.columns
    r = r.to_dict('index')
    dt = float(custom_round(custom_round(custom_round(T / n_obser),3)))
    iterations += dt
    iterations = np.round(iterations,3)
    #print(iterations)
    iteration = [0.0] + iterations.tolist()
    v=t
    P = []
    L = []
    if m==None:
        for m in range(n_traject):
            print('m_',m)
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

    else:
        #print('dt',dt)
        p_ = {}
        l_ = {}
        j = dt
        w = j
        for j in iterations:
            t = float(v)
            i = list(iteration).index(t)
            p = []
            l = []
            for t in iteration[i:]:
                w = j
                if j <= t:
                    p.append(getA(j, t, α=0.1, sigma=0.15) * math.exp(
                        -1 * BondPrice(j, t, α=0.1) * r['trajectoire_' + str(m)][round(w - dt, 3)]))
                    l.append(R - (1 / 𝜏) * ((getA(j, t - 𝜏, α=0.1, sigma=0.15) * math.exp(
                        -1 * BondPrice(j, t - 𝜏, α=0.1) * r['trajectoire_' + str(m)][round(w - dt, 3)]) / getA(j, t,
                                                                                                               α=0.1,
                                                                                                               sigma=0.15) * math.exp(
                        -1 * BondPrice(j, t, α=0.1) * r['trajectoire_' + str(m)][round(w - dt, 3)])) - 1))
            p_[j] = p
            l_[j] = l
        P.append(p_)
        L.append(l_)

    K = P
    P = pd.DataFrame(P)
    L = pd.DataFrame(L)

    return (L, P, K)

#Simulation du Vrec
#N: notional
def simulationVrec(n_traject,n_obser, N, T, r=0.03,𝜏= 0.5):
    dt = float(custom_round(custom_round(custom_round(T / n_obser),3)))
    Vrec = {}
    L, P, K = simulationP(n_traject, n_obser, T, r, 1, 𝜏,0)
    for m in range(n_traject):
        V = {}
        for t_obs in K[0].keys():
            print('pas:',t_obs)
            d=0
            for i in  [0.0]+[k+𝜏 for k in range(T)]+[float(T)]:
                #print('i',i)
                L,P,_ = simulationP(n_traject,n_obser, T,r, i, 𝜏,m)

                if L.iloc[0][t_obs]!=[] and P.iloc[0][t_obs]!=[]:
                    g = P.iloc[0][t_obs]
                    h = L.iloc[0][t_obs]
                    d+=N*(h[0]*g[0])
            V[t_obs]=d
        #print('mVrec',m)
        Vrec[m]=V
    Vrec = pd.DataFrame(Vrec).T
    return(Vrec)

def simUnitVrect(P_trajectoire,T, n_obser, dt, K=0.03, N = 1):
    Vrec = []
    for i in range(1,n_obser-1):
        P_t_obs = P_trajectoire[P_trajectoire.temps == round(i*dt,3)]
        P_t_obs = P_t_obs[P_t_obs.columns[3:]]
        if len(P_t_obs) != 0:
            P_t_obs = P_t_obs.values[0]
            L_list = getL(P_t_obs, 𝜏 = 0.5)
            V_rec_th_time = getVrec(K, L_list,P_t_obs, i*dt, T)
            Vrec.append(V_rec_th_time)
        else:
            Vrec.append("no")
    return Vrec
def simulationVrecV2(n_traject, n_obser, T, N, K = 0.03):
    P = simulationPV2(n_traject, n_obser, T)
    P.reset_index(inplace=True)
    Sim_Vrec = []
    dt = T/n_obser
    for trajectoire in P.trajectoire.unique():
        P_trajectoire = P[P.trajectoire == trajectoire]
        Sim_Vrec.append(simUnitVrect(P_trajectoire,T, n_obser,dt, K))
    return Sim_Vrec