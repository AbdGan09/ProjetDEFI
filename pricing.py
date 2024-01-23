#developpement de tout ce qui est en lien avec la partie pricing

from quant import *
from scipy import integrate
from scipy.optimize import newton
from scipy.optimize import fsolve
import scipy.optimize._minimize
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
def simulationP(n_traject, n_obser, T,R, t, 𝜏, i_traj=0,r=None, isPsimulation=False):
    if isPsimulation:
        r = simulationProcessusTaux(n_traject, n_obser, T)
        r = r.to_dict('index')

    dates_paiement = np.round(np.arange(0, (T / 𝜏)+1, 𝜏), 4)
    P = []
    L = []

    p_ = {}
    l_ = {}

    j0=0
    for j in dates_paiement[1:len(dates_paiement)-1]:
        if t<= j:
            t = float(t)

            p = zeroCoupon(t, j, r['trajectoire_'+str(i_traj)][t])
            l = R - (1 / 𝜏) * ((zeroCoupon(t, j-j0, r['trajectoire_'+str(i_traj)][t]) /zeroCoupon(t, j, r['trajectoire_' + str(i_traj)][t])) - 1)

            j0 =j
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
            L, P= simulationP(n_traject, n_obser, T, R, t_obs, 𝜏, i_traj,r)
            prix_actual = np.array(P.iloc[0]).T
            L_actual = np.array(L.iloc[0])
            v_t = np.dot(L_actual,prix_actual)
            V[t_obs] = v_t * N

        Vrec[i_traj] = V
    Vrec = pd.DataFrame(Vrec).T
    return Vrec

#CDS
def pricingCDS():
    # Fonction représentant l'intégrande pour λc
    def integrand_lambda_c(s, lambda_c):
        return np.exp(-lambda_c * s)

    # Paramètres
    RR = 0.4  # Taux de récupération
    lambda_c_constant = 0.02  # Intensité de défaut constante sur 6 mois
    T0 = 0+10e-9  # Temps initial
    T_final = 6 / 12  # Temps final en années
    T_values = [T0, 3/12, T_final]


    # Calcul final du STCDS
    def STCDS(lambda_c_constant):
        integrand = lambda s: zeroCoupon(T0, T_final, 0.03) * ((s - T0) / (T_final - T0)) * integrand_lambda_c(s,lambda_c_constant) * lambda_c_constant
        result, _ = integrate.quad(integrand, T0, T_final)

        # Calcul de la somme
        sum_term = 0.5 * sum([(zeroCoupon(T0, Ti, 0.03)*integrand_lambda_c(Ti-T0,lambda_c_constant)+integrate.quad((lambda s: zeroCoupon(T0, s, 0.03) * ((s - Tj)/(Ti - Tj)) * integrand_lambda_c(s, lambda_c_constant) * lambda_c_constant),Tj, Ti)[0]) for (Ti, Tj) in zip(T_values[1:],T_values[:-1])])

        # Print intermediate results for debugging
        #print(f"Sum term: {sum_term}")
        #print(integrate.quad((lambda s: zeroCoupon(T0, s, 0.03) * ((s - T0)/(T_final - T0)) * integrand_lambda_c(s, lambda_c_constant) * lambda_c_constant),T0, T_final)[0])

        return  (1 - RR) * ((integrate.quad((lambda s: zeroCoupon(T0, s, 0.03)*integrand_lambda_c(s,lambda_c_constant)*lambda_c_constant),T0, T_final))[0]/ sum_term)

    S_CDS = STCDS(lambda_c_constant)

    # Utiliser la méthode de Powell pour trouver numériquement la valeur de lambda_c

    lambda_c_numeric = scipy.optimize.minimize(lambda lambda_c: (STCDS(lambda_c) - (49.5 / 10000)) ** 2,lambda_c_constant - 0.01, method="Powell")
    lambda_c_numeric1 = scipy.optimize.minimize(lambda lambda_c: (STCDS(lambda_c) - (49.5/10000)) ** 2, lambda_c_constant - 0.01, method="COBYLA")

    print(f"La valeur du CDS sur l'intervalle [0, 6 mois] est : {S_CDS}")
    print(f"La valeur de lambda_c_constant est : {lambda_c_constant}")
    print(f"La valeur numérique de lambda_c par la méthode de Powell est : {lambda_c_numeric}")
    print(f"La valeur numérique de lambda_c par la méthode de Cobyla est : {lambda_c_numeric1}")
