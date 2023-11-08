#developpement de tout ce qui est en lien avec la partie pricing
#importation library

from quant import *

#fonction qui prend en param le notionnel N, la date de valo, le taux fixe du swap
def pricingSwap():
    pass


#simulation de trajectoire d'un brownien
def brownian(M,I):
    def gen_sn(M, I, anti_paths=True, mo_match=True):
        ''' 
        Paramètres
        ==========
        M: int
        nombre d'intervalles temporels de discrétisation
        I: int
        nombre de trajectoires à simuler
        anti_paths: boolean
        usage de variables antithétiques
        mo_match: boolean
        usage de l'association des moments
        '''
        if anti_paths is True:
            sn = npr.standard_normal((M + 1, int(I / 2)))
            sn = np.concatenate((sn, -sn), axis=1)
        else:
            sn = npr.standard_normal((M + 1, I))
        if mo_match is True:
            sn = (sn - sn.mean()) / sn.std()
        return sn
    brownian_motion = np.cumsum(gen_sn(M,I), axis=0)
    return brownian_motion


#Simulation du processus
def simulationProcessusTaux():
    pass


#Simulation du discount
def simulationP():
    pass


#Simulation du Vrec
def simulationVrect():
    pass
