from mpmath import mp
import numpy as np
import matplotlib.pyplot as plt
mp.dps = 500

def DimensionlessTemperature(Da, Cv, Br, points):
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

    YArray = [None]*(points)
    tfArray = [None]*points
    tsArray = [None]*points
    tArray = [None]*points
    Y = mp.zero
    i = 0
    step = 1/mp.mpf(points - 1)
    
    while Y < 1:
        tf =  C3 + C4*Y + C5*Y**2 + A1*mp.cosh(Y*G) - C7*mp.cosh((1 - 2*Y)/mp.sqrt(Da))*mp.csch(1/mp.sqrt(Da)) + C9*mp.cosh((1 - 2*Y)/(2*mp.sqrt(Da)))*mp.sech(1/(2*mp.sqrt(Da))) + A2*mp.sinh(Y*G)
        ts =  -T4 - T5 + T2*Y + T3*Y**2 - T6*mp.cosh((1 - 2*Y)/mp.sqrt(Da))*mp.csch(1/mp.sqrt(Da)) + T5*mp.cosh((1 - 2*Y)/(2*mp.sqrt(Da)))*mp.sech(1/(2*mp.sqrt(Da))) - Cv * tf
        t =  -(P*(2*Da**2 + Da*(-2 + Y)*Y + Br*P*(1 + P)*(-2 + Y)*Y + 2*Br*Da*P*(2 + P + 2*P*Y) + (2*Da**2 + Br*Da*P*(3 + P*(2 - 4*Y)) + Da*(-2 + Y)*Y + Br*P*(1 + P)*(-2 + Y)*Y)*mp.cosh(1/mp.sqrt(Da)) + Br*Da*P*mp.cosh((1 - 2*Y)/mp.sqrt(Da)) - 2*Da*(Da + Br*P*(2 + P))*mp.cosh((1 - Y)/mp.sqrt(Da)) - 2*mp.sqrt(Da)*(-(Da*Y) + Br*P*(2*Da*P - Y + P*(-3 + Y)*Y))*mp.sinh(1/mp.sqrt(Da)) - 2*Da*mp.cosh(Y/mp.sqrt(Da))*(Da + Br*P*(2 + P) - 2*Br*mp.sqrt(Da)*P**2*mp.sinh(1/mp.sqrt(Da))) - 8*Br*Da**mp.mpf('1.5')*P**2*mp.sinh(1/(2*mp.sqrt(Da)))**2*mp.sinh(Y/mp.sqrt(Da))))/(2*(1 + Cv)*Da*(1 + mp.cosh(1/mp.sqrt(Da))))
        
        tfArray[i] = float(tf)
        tsArray[i] = float(ts)
        tArray[i] = float(t)
        YArray[i] = float(Y)
        Y += step
        i += 1
    
    if i < points:
        Y=mp.one
        tf =  C3 + C4*Y + C5*Y**2 + A1*mp.cosh(Y*G) - C7*mp.cosh((1 - 2*Y)/mp.sqrt(Da))*mp.csch(1/mp.sqrt(Da)) + C9*mp.cosh((1 - 2*Y)/(2*mp.sqrt(Da)))*mp.sech(1/(2*mp.sqrt(Da))) + A2*mp.sinh(Y*G)
        ts =  -T4 - T5 + T2*Y + T3*Y**2 - T6*mp.cosh((1 - 2*Y)/mp.sqrt(Da))*mp.csch(1/mp.sqrt(Da)) + T5*mp.cosh((1 - 2*Y)/(2*mp.sqrt(Da)))*mp.sech(1/(2*mp.sqrt(Da))) - Cv * tf
        t =  -(P*(2*Da**2 + Da*(-2 + Y)*Y + Br*P*(1 + P)*(-2 + Y)*Y + 2*Br*Da*P*(2 + P + 2*P*Y) + (2*Da**2 + Br*Da*P*(3 + P*(2 - 4*Y)) + Da*(-2 + Y)*Y + Br*P*(1 + P)*(-2 + Y)*Y)*mp.cosh(1/mp.sqrt(Da)) + Br*Da*P*mp.cosh((1 - 2*Y)/mp.sqrt(Da)) - 2*Da*(Da + Br*P*(2 + P))*mp.cosh((1 - Y)/mp.sqrt(Da)) - 2*mp.sqrt(Da)*(-(Da*Y) + Br*P*(2*Da*P - Y + P*(-3 + Y)*Y))*mp.sinh(1/mp.sqrt(Da)) - 2*Da*mp.cosh(Y/mp.sqrt(Da))*(Da + Br*P*(2 + P) - 2*Br*mp.sqrt(Da)*P**2*mp.sinh(1/mp.sqrt(Da))) - 8*Br*Da**mp.mpf('1.5')*P**2*mp.sinh(1/(2*mp.sqrt(Da)))**2*mp.sinh(Y/mp.sqrt(Da))))/(2*(1 + Cv)*Da*(1 + mp.cosh(1/mp.sqrt(Da))))

        tfArray[i] = float(tf)
        tsArray[i] = float(ts)
        tArray[i] = float(t)
        YArray[i] = float(Y)
    
    return (YArray, tfArray, tsArray, tArray)

Y, tf, ts, t = DimensionlessTemperature(Da = 0.001, Cv = 0.01, Br = 1, points = 1000)

plt.plot(Y, tf, label='θf')
plt.plot(Y, ts, label='θs')
plt.plot(Y, t, label='θ')
plt.legend()
plt.show()