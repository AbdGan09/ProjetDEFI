#lieu pour mettre les class ou fonction de test:
from pricing import *
from quant import *
data = importData(r'boostrapping_etudiants2.xlsx', "Donnee")
spread_CDS = importData(r'spreads_CDS.xlsx', 'SpreadsCDS')
cs = ZeroCouponCurve(data)
class Main:
    def __init__(self, param):
        pass

    @staticmethod
    def test_Hull_White():
        donne_simule = hullWhite(True, list(np.random.normal(0, 1, 1000)))
        t_liste = np.linspace(0, 1, 1000)
        print(len(donne_simule))
        print(len(t_liste))
        plotSimulation(t_liste, donne_simule)

    @staticmethod
    def HullWhite2(isForSimulation = False, dW = None, tau = 0.01, tn = 1):
        k = hullWhite(isForSimulation, dW, tau, tn)
        print(k)

    @staticmethod
    def testTrajectoire(n_traject=4,n_obser=1000,T=30):
        n_obser = n_obser*T
        trajectoires = generateurTrajectoire(n_traject, n_obser,T, 𝜏= 0.5)
        print(trajectoires.head(2))
        plotSimulation(trajectoires.columns, list(trajectoires.T.values))

    @staticmethod
    def testSimulationTaux(n_traject=4, n_obser=100, T=10):
        n_obser = T*n_obser
        simulation_Taux = simulationProcessusTaux(n_traject, n_obser, T, True)
        print(simulation_Taux)
        plotSimulation(simulation_Taux.columns, list(simulation_Taux.T.values))

    def testSimulationP(n_traject=1, n_obser=100, T=10,R=0.03, t=4.5, 𝜏=0.5, r=None,isPsimulation=True):
        L,simulation_P = simulationP(n_traject,n_obser, T,R, t, 𝜏, r=None,isPsimulation=True)
        print(simulation_P)
        print(L)
        #simulation_P = pd.DataFrame(simulation_P)
        #plotSimulation(simulation_P.columns, list(simulation_P.T.values))
    def testSimulationVrec(n_traject=10,n_obser=3000, N=100, T=30, R=0.03,𝜏= 0.5):
        simulation_Vrec = simulationVrec(n_traject,n_obser, N, T, R=0.03,𝜏= 0.5)
        print(simulation_Vrec.mean(axis=0))
        print(simulation_Vrec)
        simulation_Vrec.T.plot()
        simulation_Vrec.mean(axis=0).T.plot()
        plt.show()
        #simulation_Vrec = pd.DataFrame(simulation_Vrec)
        #plotSimulation(simulation_Vrec.columns, list(simulation_Vrec.values))

    def courbe_fwdinst(t, α=0.1, sigma=0.15):
        x = np.linspace(1,t,100)
        print(x)
        courbe = []
        for j in x:
            courbe.append(MarketZeroCoupon.getmarketZeroCouponInstFwdCurve(j))
        print(courbe)
        plt.plot(x,courbe)
        plt.show()
    def testCDS():
        pricingCDS()
    @staticmethod
    def testSixMonth():
        print("SixMonth Spread is:", calcul_SpreadCDS(0.0025, 0.5, cs, spread_CDS, 𝜏i=0.25, RR=0.4, T0=0))
        print("Defaut Intensity for 6 Month is",get_Default_Intensity(12.32, 0.5, cs, spread_CDS, lambda_c_constant=0.0001))

    @staticmethod
    def testOneYearCDS():
        print("One_Year_SpreadCDS is", One_Year_SpreadCDS(0.04, 1, cs, spread_CDS, 𝜏i=0.25, RR=0.4, T0=0))
        print("Defaut Intensity for On Year is",
              get_Default_Intensity(46.5, 1, cs, spread_CDS, lambda_c_constant=0.0001))

    @staticmethod
    def testNyearCDS(N=10):
        res = SpreadCDSRecursive(0.04, N, cs, spread_CDS, 𝜏i=0.25, RR=0.4, T0=0)
        Y = list(res.values())
        Y.append(Y[-1])
        X = list(res.keys())
        X.insert(0,0)
        plt.step(X,Y, where='post', linestyle='-')
        plt.show()

    def testEPE():
        E = calcul_EPE(10, 15, n_traject=4, n_obser=3000, N=100, T=30, R=0.03, 𝜏=0.5)
        print (E)

    def testCVA(N=10):
        dict_lambdas = SpreadCDSRecursive(0.04, N, cs, spread_CDS, 𝜏i=0.25, RR=0.4, T0=0)
        CVA = calcul_CVA(0, 10, dict_lambdas)
        print ('La CVA est: ',CVA)

#Main.courbe_fwdinst(10)
#Main.test_Hull_White()
#Main.testTrajectoire()
#Main.testSimulationTaux()
#Main.testSimulationP()
#Main.testSimulationVrec()
#Main.testCDS()

#Main.testSixMonth()
#Main.testOneYearCDS()
#Main.testNyearCDS()

#Main.testEPE()
Main.testCVA()