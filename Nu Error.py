from multiprocessing import Pool
from mpmath import mp
import numpy as np
mp.dps = 1000

def Nu(Da, Cv, Br):
    P = 1/(-1 + 2 * mp.sqrt(Da) * mp.tanh(1/(2 * mp.sqrt(Da))))

    a = mp.sqrt(mp.one/(mp.mpf('12')*Da))
    Nu = mp.mpf('2.253') + mp.mpf('8.164') * (a/(a + mp.one))**mp.mpf('1.5')
    Dv = Cv * Nu * a * (a + mp.mpf('1'))
    G = mp.sqrt(Dv*(1 + Cv)) / mp.sqrt(Cv)

    T2 = -((P*(-1 + mp.sqrt(Da) * mp.tanh(1/(2 * mp.sqrt(Da))))*(Da + Br * P * (1 + P) - 2 * Br * mp.sqrt(Da) * P**2 * mp.tanh(1/(2 * mp.sqrt(Da)))))/Da)
    T3 = -(P*(Da + Br*P*(1 + P) - 2*Br*mp.sqrt(Da)*P**2*mp.tanh(1/(2*mp.sqrt(Da)))))/(2*Da)
    T4 = -(Br*P**2*mp.cosh(1/mp.sqrt(Da))*mp.sech(1/(2*mp.sqrt(Da)))**2)/4
    T5 = P*(Da + Br*P*(2 + P) - 2*Br*mp.sqrt(Da)*P**2*mp.tanh(1/(2*mp.sqrt(Da))))
    T6 = (Br*P**2*mp.tanh(1/(2*mp.sqrt(Da))))/2

    C3 = -((2*T3 - Dv*(-T4 - T5) - Cv*Dv*(-T4 - T5))/((1 + Cv)**2*Dv))
    C4 = T2/(1 + Cv)
    C5 = T3/(1 + Cv)
    C7 = ((-4 + Da*Dv)*T6)/(-4*Cv + Da*Dv + Cv*Da*Dv)
    C9 = -((T5 - Da*Dv*T5)/(-Cv + Da*Dv + Cv*Da*Dv))

    A1 = -C3 - C9 + C7*mp.coth(1/mp.sqrt(Da))
    A2 = (mp.sech(G)*(2*C7 - (C9 + 2*C5*Da)*mp.tanh(1/(2*mp.sqrt(Da)))))/(mp.sqrt(Da)*G) + (C3 + C9 - C7*mp.coth(1/mp.sqrt(Da)))*mp.tanh(G)

    # tfb = (P*(-6*A2 + (6*C3 + 3*C4 + 2*(C5 - 3*C7*mp.sqrt(Da) + 6*C5*Da))*G*(1 - Da*G**2) - (2*G*(-1 + Da*G**2)*(-3*C9 + 4*C7*mp.sqrt(Da) + 2*C7*mp.sqrt(Da)*mp.cosh(1/mp.sqrt(Da))))/(1 + mp.cosh(1/mp.sqrt(Da))) + 6*A2*mp.cosh(G) + 6*A1*mp.sinh(G) - 6*mp.sqrt(Da)*G*(A1 - (2*C3 + C4 + C5 - C9 + 4*C5*Da)*(-1 + Da*G**2) + A1*mp.cosh(G) + A2*mp.sinh(G))*mp.tanh(1/(2*mp.sqrt(Da)))))/(6*G*(-1 + Da*G**2))
    tfb = (C5*Da*P*G**2)/(3 - 3*Da*G**2) + (C4*Da*P*G**2)/(2 - 2*Da*G**2) + (C7*mp.sqrt(Da)*P)/(1 - Da*G**2) + (C3*Da*P*G**2)/(1 - Da*G**2) + (C3*P)/(-1 + Da*G**2) + (2*C5*Da*P)/(-1 + Da*G**2) +  (C7*Da**mp.mpf('1.5')*P*G**2)/(-1 + Da*G**2) - (2*C5*Da**2*P*G**2)/(-1 + Da*G**2) + (C4*P)/(-2 + 2*Da*G**2) + (C5*P)/(-3 + 3*Da*G**2) + (A2*P)/(G - Da*G**3) -  (C9*P)/((-1 + Da*G**2)*(1 + mp.cosh(1/mp.sqrt(Da)))) + (4*C7*mp.sqrt(Da)*P)/(3*(-1 + Da*G**2)*(1 + mp.cosh(1/mp.sqrt(Da)))) + (C9*Da*P*G**2)/((-1 + Da*G**2)*(1 + mp.cosh(1/mp.sqrt(Da)))) -  (4*C7*Da**mp.mpf('1.5')*P*G**2)/(3*(-1 + Da*G**2)*(1 + mp.cosh(1/mp.sqrt(Da)))) + (2*C7*mp.sqrt(Da)*P*mp.cosh(1/mp.sqrt(Da)))/(3*(-1 + Da*G**2)*(1 + mp.cosh(1/mp.sqrt(Da)))) -  (2*C7*Da**mp.mpf('1.5')*P*G**2*mp.cosh(1/mp.sqrt(Da)))/(3*(-1 + Da*G**2)*(1 + mp.cosh(1/mp.sqrt(Da)))) + (A2*P*mp.cosh(G))/(-G + Da*G**3) + (A1*P*mp.sinh(G))/(-G + Da*G**3) +  (A1*mp.sqrt(Da)*P*mp.tanh(1/(2*mp.sqrt(Da))))/(1 - Da*G**2) + (C4*mp.sqrt(Da)*P*mp.tanh(1/(2*mp.sqrt(Da))))/(1 - Da*G**2) + (C5*mp.sqrt(Da)*P*mp.tanh(1/(2*mp.sqrt(Da))))/(1 - Da*G**2) +  (C9*Da**mp.mpf('1.5')*P*G**2*mp.tanh(1/(2*mp.sqrt(Da))))/(1 - Da*G**2) - (2*C3*mp.sqrt(Da)*P*mp.tanh(1/(2*mp.sqrt(Da))))/(-1 + Da*G**2) + (C9*mp.sqrt(Da)*P*mp.tanh(1/(2*mp.sqrt(Da))))/(-1 + Da*G**2) -  (4*C5*Da**mp.mpf('1.5')*P*mp.tanh(1/(2*mp.sqrt(Da))))/(-1 + Da*G**2) + (2*C3*Da**mp.mpf('1.5')*P*G**2*mp.tanh(1/(2*mp.sqrt(Da))))/(-1 + Da*G**2) + (C4*Da**mp.mpf('1.5')*P*G**2*mp.tanh(1/(2*mp.sqrt(Da))))/(-1 + Da*G**2) +  (C5*Da**mp.mpf('1.5')*P*G**2*mp.tanh(1/(2*mp.sqrt(Da))))/(-1 + Da*G**2) + (4*C5*Da**mp.mpf('2.5')*P*G**2*mp.tanh(1/(2*mp.sqrt(Da))))/(-1 + Da*G**2) + (A1*mp.sqrt(Da)*P*mp.cosh(G)*mp.tanh(1/(2*mp.sqrt(Da))))/(1 - Da*G**2) +  (A2*mp.sqrt(Da)*P*mp.sinh(G)*mp.tanh(1/(2*mp.sqrt(Da))))/(1 - Da*G**2)
    return -(2/(Cv * tfb))

def ENu(Da, Cv, Br):
    nu0 = Nu(Da, Cv, 0)
    nu = Nu(Da, Cv, Br)
    return (nu - nu0)/nu

def data(i):
    lDa, lCv, Br = i
    Da = 10**lDa
    Cv = 10**lCv
    return ['%.8f'%lDa, '%.8f'%lCv, '%.8f'%ENu(Da, Cv, Br)]

if __name__ == '__main__':
    with Pool(8) as p:
        lDaMin = mp.mpf('-5')
        lDaMax= mp.mpf('4')
        lDaStep = mp.mpf('0.05')
        lDaSize = (lDaMax - lDaMin)/lDaStep + 1

        lCvMin = mp.mpf('-3')
        lCvMax= mp.mpf('0')
        lCvStep = mp.mpf('0.05')
        lCvSize = (lCvMax - lCvMin)/lCvStep + 1
        
        lDaArray = [None]*int(lDaSize)
        lCvArray = [None]*int(lCvSize)

        for i in range(int(lDaSize)):
            lDaArray[i] = lDaMin + i * lDaStep
        
        for i in range(int(lCvSize)):
            lCvArray[i] = lCvMin + i * lCvStep
        
        Br = '0.00'
        ans = p.map(data, [(lDa, lCv, mp.mpf(Br)) for lDa in lDaArray for lCv in lCvArray])
        with open("Br=" + Br + ".txt", "a") as f:
            print('\n'.join(['\t'.join([str(j) for j in i])  for i in ans]), file=f)
    
