#developpement de tout ce qui est en lien avec la partie pricing

from quant import *
from scipy import integrate
from scipy.optimize import newton
from scipy.optimize import fsolve
import scipy.optimize._minimize
#simulation de trajectoire
data = importData(r'boostrapping_etudiants2.xlsx', "Donnee")
spread_CDS = importData(r'spreads_CDS.xlsx', 'SpreadsCDS')
cs = ZeroCouponCurve(data)
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

# Calcul de l'EPE
def calcul_EPE(t,S,n_traject=10,n_obser=3000, N=100, T=30, R=0.03,𝜏= 0.5):
    Vrec = simulationVrec(n_traject, n_obser, N, T, R=0.03, 𝜏=0.5)
    Vrec = Vrec.fillna(0)
    RR = 0.03
    LGD = 1-RR
    if t==0:
        D = zeroCoupon(t+0.0001, S, 0.03)
    else:
        D = zeroCoupon(t, S, 0.03)
    valeur = np.maximum(Vrec.loc[:,t:S],0)
    EPE = valeur.mean(axis=1)*LGD*D
    return(EPE)
def calcul_CVA(t,T,dico_lambdas):
    mat = list(dico_lambdas.keys())
    mat.insert(0,'0')
    if float(T)==float(mat[-1]):
        CVA = 0
        for i in range(1,len(mat)):
            mat_prime = mat[1:i+1]
            e = 1
            for j in range(len(mat_prime)):
                e*=integrand_lambda_c(float(mat_prime[j]), dico_lambdas[mat_prime[j]])
            EPE = calcul_EPE(t,float(mat[i]),n_traject=4,n_obser=1000, N=100, T=10, R=0.03,𝜏= 0.5).mean()
            CVA+=(float(mat[i])-float(mat[i-1]))*dico_lambdas[mat[i]]*e*EPE
        return(CVA)

    else:
        return('erreur')

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

        return  (1 - RR) * ((integrate.quad((lambda s: zeroCoupon(T0, s, 0.03)*integrand_lambda_c(s,lambda_c_constant)*lambda_c_constant),T0, T_final))[0]/ sum_term)

    S_CDS = STCDS(lambda_c_constant)

    # Utiliser la méthode de Powell pour trouver numériquement la valeur de lambda_c

    lambda_c_numeric = scipy.optimize.minimize(lambda lambda_c: (STCDS(lambda_c) - (12.32 / 10000)) ** 2,lambda_c_constant - 0.01, method="Powell")
    lambda_c_numeric1 = scipy.optimize.minimize(lambda lambda_c: (STCDS(lambda_c) - (12.32/10000)) ** 2, lambda_c_constant - 0.01, method="COBYLA")

    print(f"La valeur du CDS sur l'intervalle [0, 6 mois] est : {S_CDS}")
    print(f"La valeur de lambda_c_constant est : {lambda_c_constant}")
    print(f"La valeur numérique de lambda_c par la méthode de Powell est : {lambda_c_numeric}")
    print(f"La valeur numérique de lambda_c par la méthode de Cobyla est : {lambda_c_numeric1}")


#CDS 6months and CDS 1year
def get_Default_Intensity(spread, Maturity, ZC_curve, spreads_data=None, lambda_c_constant=0.0001):
    lambda_c_numeric = scipy.optimize.minimize(lambda lambda_c: (NYear_SpreadCDS(lambda_c, Maturity, ZC_curve, spreads_data) - (spread / 10000)) ** 2, 0.0008,bounds=[(0.0006,0.03)], method="Powell").x[0]
    return lambda_c_numeric


def calcul_SpreadCDS(lambdas, Maturity, ZC_curve, spreads=None, 𝜏i=0.25, RR=0.4, T0=0):
    # pour 6mois:
    integrand_deno = lambda s: ZC_curve(s) * ((s - T0) / (𝜏i)) * integrand_lambda_c(s, lambdas) * lambdas
    term_deno = 0
    for i in range(int(Maturity / 𝜏i) + 1):
        first_term = ZC_curve((i + 1) * 𝜏i) * integrand_lambda_c((i + 1) * 𝜏i, lambdas)
        # print("first_term ", first_term)
        second_term, _ = integrate.quad(integrand_deno, i * 𝜏i, (i + 1) * 𝜏i)
        # print("second_term ",second_term)
        term_deno += (first_term + second_term) * 𝜏i
    term_num = (1 - RR) * (integrate.quad((lambda s: ZC_curve(s) * integrand_lambda_c(s, lambdas) * lambdas), T0, Maturity))[0]
    return term_num / term_deno

def One_Year_SpreadCDS(lambdas, Maturity, ZC_curve, spreads_data, 𝜏i=0.25, RR=0.4, T0=0):
    # pour 1 year je dois splité mes intégrale ens deux à chaque fois pour tenir conte de la constance entre 0 et 6 et entre 6 et 12 mois résiduelle
    integrand_deno = lambda s: ZC_curve(s) * ((s - T0) / (𝜏i)) * integrand_lambda_c(s, lambdas) * lambdas
    term_deno = 0
    for i in range(int(Maturity / 𝜏i) + 1):
        current_lambda = lambdas
        if (i + 1) * 𝜏i <= 0.5:
            current_lambda = lambda_6_M
            integrand_deno_6M = lambda s: ZC_curve(s) * ((s - T0) / (𝜏i)) * integrand_lambda_c(s, current_lambda) * current_lambda
            first_term = ZC_curve((i + 1) * 𝜏i) * integrand_lambda_c((i + 1) * 𝜏i, current_lambda)
            second_term, _ = integrate.quad(integrand_deno_6M, i * 𝜏i, (i + 1) * 𝜏i)
            term_deno += (first_term + second_term) * 𝜏i
        else:
            first_term = ZC_curve((i + 1) * 𝜏i) * (integrand_lambda_c(0.5, lambda_6_M) + integrand_lambda_c(((i + 1) * 𝜏i - 0.5), lambdas))
            second_term, _ = integrate.quad(integrand_deno, i * 𝜏i, (i + 1) * 𝜏i)
            term_deno += (first_term + second_term) * 𝜏i
    term_num = (1 - RR) * ((integrate.quad((lambda s: ZC_curve(s) * integrand_lambda_c(s, lambda_6_M) * lambda_6_M), T0, 0.5))[0] + (integrate.quad((lambda s: ZC_curve(s) * (integrand_lambda_c(s, lambdas)) * lambdas), 0.5, Maturity))[0])
    return term_num / term_deno

def NYear_SpreadCDS(lambdas, Maturity, ZC_curve, lambda_data, 𝜏=0.25, RR=0.4, T0=0):
    lambda_6_M = scipy.optimize.minimize(lambda lambda_c: (calcul_SpreadCDS(lambda_c, spread_CDS["Matu_By_Year"][0], ZC_curve, spread_CDS) - (spread_CDS["VOWG6MEUAM=R"][0] / 10000)) ** 2, 0.0008, bounds=[(0.0006,0.03)],method="Powell").x[0]
    term_deno = 0
    term_num = (integrate.quad((lambda s: ZC_curve(s) * integrand_lambda_c(s, lambda_6_M) * lambda_6_M), T0, 0.5))[0]
    mat = [0.5, 1, 2, 3,4, 5,7,10]
    i=0
    current_integrant = integrand_lambda_c(0.5, lambda_6_M)
    while i <=(Maturity / 𝜏):
        if (i + 1) * 𝜏 <= 0.5:
            current_lambda = lambda_6_M
            integrand_deno_6M = lambda s: ZC_curve(s) * ((s - (i*𝜏)) / (𝜏)) * integrand_lambda_c(s, current_lambda) * current_lambda
            first_term = ZC_curve((i + 1) * 𝜏) * integrand_lambda_c((i + 1) * 𝜏, current_lambda)
            second_term, _ = integrate.quad(integrand_deno_6M, i * 𝜏, (i + 1) * 𝜏)
            term_deno += (first_term + second_term) * 𝜏
            i+=1
        else:
            current_integrant *= integrand_lambda_c((𝜏), current_lambda)
            integrand_deno = lambda s: ZC_curve(s) * ((s - i * 𝜏) / (𝜏)) * integrand_lambda_c(s, current_lambda) * current_integrant * current_lambda
            try:
                while (i + 1) * 𝜏 <= mat[1]:
                    current_lambda = lambda_data[str(float(mat[1]))]
                    first_term *= ZC_curve((i + 1) * 𝜏) * (integrand_lambda_c((i + 1) * 𝜏 - i * 𝜏, current_lambda))
                    second_term *= integrate.quad(integrand_deno, i * 𝜏, (i + 1) * 𝜏)[0]
                    i+=1
                term_deno += (first_term + second_term) * 𝜏
                term_num = sum([integrate.quad((lambda s: ZC_curve(s) * (integrand_lambda_c(s, lambda_data[str(float(mat[i]))])) * lambda_data[str(float(mat[i]))]), mat[i],mat[i + 1])[0] for i in range(len(mat))])
                mat = mat[1:]
            except:
                current_lambda = lambdas
                integrand_deno = lambda s: ZC_curve(s) * ((s - i * 𝜏) / (𝜏)) *  integrand_lambda_c(s,current_lambda,mat[0]) * current_lambda
                first_term  *= ZC_curve((i + 1) * 𝜏) * (integrand_lambda_c((i + 1) * 𝜏 - i * 𝜏, current_lambda))
                second_term *= integrate.quad(integrand_deno, i * 𝜏, (i + 1) * 𝜏)[0]
                i+=1
            term_deno += (first_term + second_term) * 𝜏
            term_num += (integrate.quad((lambda s: ZC_curve(s) * integrand_lambda_c(s, lambdas) * lambdas), mat[0],mat[1]))[0]

    return ((1 - RR) * term_num )/ term_deno

def SpreadCDSRecursive(lambdas, Maturity, ZC_curve, spread_CDS, 𝜏i=0.25, RR=0.4, T0=0):
    dico_Lambda = {}
    index_matu = list(spread_CDS["Matu_By_Year"]).index(Maturity)
    for i in range(index_matu+1):
        dico_Lambda[str(spread_CDS["Matu_By_Year"][i])] = get_Default_Intensity(spread_CDS["VOWG6MEUAM=R"][i], spread_CDS["Matu_By_Year"][i], ZC_curve, dico_Lambda)
    print('dico_lambda',dico_Lambda)
    return (dico_Lambda)