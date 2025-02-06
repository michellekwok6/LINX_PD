import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid as trapz
from scipy.integrate import dblquad, quad
from linx.const import aFS, me, eta0, Emin, Ephb_T_max, NE_pd, NE_min, eps, approx_zero
from linx.special_funcs import zeta_3
from quadax import quadgk
import time

#Abundances at the end of BBN given relative to baryons ie Y_N = n_N/n_b
acro_Y = {
"n" : 0.000000e+00, 
"p" : 7.525079e-01,
"d" : 1.837924e-05,
"t" : 5.812933e-08,
"He3" : 7.697251e-06,
"He4" : 6.185801e-02,
"Li6" : 8.104749e-15,
"Li7" : 2.157596e-11,
"Be7" : 3.871693e-10
}


class InjectedSpectrum(eqx.Module):

    """
    Injected spectrum of e+/e- or photons. In units of MeV^2 in acropolis
    Includes the following processes:
        1. Double photon pair creation: (g + g_thermal -> e+ + e-
        2. Photon photon scattering: (g + g_thermal -> g + g)
        3. Bethe-Heitler pair creation: (g + N -> e+ e- + N where N is p or He4)
        4. Computon Scattering: (g + e-_thermal -> g + e-) 
        5. Inverse Compton scattering: (e+- + g_thermal -> e+- + g)
    Follows Hufnagel 2018
    The "JIT" functions are at the bottom of the page
    """
    def __init__(self):
        pass


    ##############
    #Photon Rates#
    ##############

    @eqx.filter_jit
    def dphoton_pair_prod_rate(self, E, T):
        """
        TODO: make all these integral compatible with jax

        Double photon pair production. 
        From equation B.6 in Hufnagel 2018

        Parameters
        ----------
        E: float
            Energy of outgoing particle (MeV)

        T: float
            Temperature of the thermal bath (MeV)

        Returns
        -------
        float
            Gamma^DP_photon (MeV)

        """

        #Check if incoming energy is greater than me^2/(22T)
        #acropolis uses 50T for a smaller threshold
        #if E < (me**2)/(50*T):
        #    return 0

        #rewritten to be with quadgk, since it is compatible with jax
        @jax.jit
        def inside_int(s):
            s = jnp.exp(s)
            b = jnp.sqrt(1 - (4*me**2)/s)
            dp_cross_section = jnp.pi * aFS**2 /(2*me**2) * (1-b**2) * ((3-b**4) * jnp.log((1+b)/(1-b)) - 2*b *(2-b**2))  
            return dp_cross_section * s * s  

        @jax.jit
        def integral_of_inside(ep):
            return quadgk(inside_int, [jnp.log(4*me**2), jnp.log(4*E* ep)], epsrel = eps, epsabs=0)[0]

        @jax.jit
        def outside_integral(ep, T):
            ep = jnp.exp(ep)
            return integral_of_inside(ep) * f_thermal_photon_spec(ep, T)/ep


        ep_ll = jnp.log((me**2)/E)
        ep_ul = jnp.log(Ephb_T_max*T)

        #I_dp_pp = 1/(8*E**2) * dblquad(dpr_integrand, ep_ll, ep_ul, lambda x: jnp.log(4*me**2), lambda x: jnp.log(4*E) + x, epsrel=eps, epsabs=0, args=(T,))[0]
        I_dp_pp = 1/(8*E**2) * quadgk(outside_integral, [ep_ll, ep_ul], epsrel=eps, epsabs=0, args=(T,))[0]

        #return I_dp_pp
        return jnp.select([E < (me**2)/(50*T)], [0], default=I_dp_pp)
    
    @eqx.filter_jit
    def photon_photon_scattering_rate(self, E, T):
        """
        Photon-photon scattering. 
        From equation B.9 in Hufnagel 2018
        """

        #Doesn't really matter since the other rates dominate in for larger E anyway so the paper puts in an exp factor.
        #if E > (me**2)/(T): 
        #    return 0
        
        #TODO:Look at the exponential decay. It's not anywhere where they originally derived it or in the paper?
        exp_fac = jnp.exp(-E*T/me**2)

        return 15568*(jnp.pi**3)/(3189375) * aFS**4 * me * (E/me)**3 * (T/me)**6 * exp_fac

    @eqx.filter_jit
    def bethe_heitler_pair_prod_rate(self, E, T):
        """
        Bethe Heitler pair production rate 
        Equation B.11 in Hufnagel 2018 and equation 34-36 in Kawasaki 1995 
        """

        k = E/me


        #Equation 34 Kawasaki 1995 
        #It can also be approximated with a constant but this is not done for a smooth transition for k

        term1 = aFS**3 /me**2 * Z2n(T)
        #if 2 <= k < 4:
        rho = (2*k - 4)/(k + 2 + 2*jnp.sqrt(2*k))
        term2 = ((k-2)/k)**3 * (1 + rho/2 + 23/40*rho**2 + 11/60*rho**3 + 29/960*rho**4)
        #if k >= 4:
        term3 = 28/9*jnp.log(2*k) - 218/27 \
        + (2/k)**2 * (2/3*jnp.log(2*k)**3 - jnp.log(2*k)**2 + (6-(jnp.pi**2)/3)*jnp.log(2*k) + 2*zeta_3 + (jnp.pi**2)/6 - 7/2) \
        - (2/k)**4 * (3/16 * jnp.log(2*k) + 1/8) \
        + (2/k)**6 * (29/2304*jnp.log(2*k) - 77/13824 )
        
        '''
        if k < 2:
            return 0
        
        if 2 <= k < 4:
            rho = (2*k - 4)/(k + 2 + 2*jnp.sqrt(2*k))
            term2 = ((k-2)/k)**3 * (1 + rho/2 + 23/40*rho**2 + 11/60*rho**3 + 29/960*rho**4)
            return term1 * term2
        

        #Equation 36 Kawasaki 1995
        if k >= 4:
            #up to order (E/me)^6
            #note: there is a typo in the Acropolis paper (Depta 2020) where the term 2/3*jnp.log(2*k)**3 is instead 2/3*jnp.log(2*k)**2
            #Checking with Hufnagel 2018 and Kawasaki 1995, as well as the acropolis code, the exponent is 3
            term2 = 28/9*jnp.log(2*k) - 218/27 \
            + (2/k)**2 * (2/3*jnp.log(2*k)**3 - jnp.log(2*k)**2 + (6-(jnp.pi**2)/3)*jnp.log(2*k) + 2*zeta_3 + (jnp.pi**2)/6 - 7/2) \
            - (2/k)**4 * (3/16 * jnp.log(2*k) + 1/8) \
            + (2/k)**6 * (29/2304*jnp.log(2*k) - 77/13824 )

            return term1 * term2
        '''
        return jnp.select([k < 2, (2 <= k) & (k < 4), k >= 4], [0, term1 * term2, term1 * term3])
    
    @eqx.filter_jit
    def compton_scattering_rate(self, E, T):
        """Compton Scattering Rate Eq. B.13 in Hufnagel 2018

        """

        x = 2*E/me

        return 2 * jnp.pi * aFS**2 /(me**2) * thermal_electron(T) * 1/x \
                * ((1 - 4/x - 8/x**2)*jnp.log(1 + x) + 1/2 + 8/x - 1/(2*(1+x)**2))
    
    @eqx.filter_jit
    def total_rate_photon(self, E, T):
        """
        Returns the total interaction rate for photons

        Parameters
        ----------
        E: float
            Energy of outgoing particle (MeV)
        T: float
            Temperature of background photons (MeV)

        Returns
        -------
        Gamma_photon (MeV)
        """
        return self.dphoton_pair_prod_rate(E, T) + self.photon_photon_scattering_rate(E, T) + self.bethe_heitler_pair_prod_rate(E, T) + self.compton_scattering_rate(E, T)
    
    
    ################
    #Photon Kernels#
    ################

    #has kernels for end state being photons

    @eqx.filter_jit
    def photon_photon_scattering_kernel(self, E, T, Ep):
        """
        Returns the total integration for photon spectrum

        Parameters
        ----------
        E: float
            Energy of outgoing particle (MeV)
        T: float
            Temperature of background photons (MeV)
        Ep: float
            Energy of incoming particle (MeV)

        Returns
        -------
        K_PP_photon (dimensionless)
        """

        exp_fac = jnp.exp(-Ep*T/me**2)
        #also the exp rate thing needs to be checked
        return 8896/(637875*jnp.pi) * (aFS**4)/me**8 * jnp.pi**4 * T**6 * Ep**2 *(1 - E/Ep + (E/Ep)**2)**2 * exp_fac

    @eqx.filter_jit
    def compton_scattering_kernel_photon(self, E, T, Ep):
        """Compton Scattering Kernel Eq. B.18 in Hufnagel 2018
            K^CS_g->g
        """
        csk = jnp.pi* (aFS**2)/(me) * thermal_electron(T)/Ep**2 * (Ep/E + E/Ep +(me/E - me/Ep)**2 - 2 * me * (1/E - 1/Ep))

        return jnp.select([Ep/(1+2*Ep/me) < E], [csk], default=0)
    
    '''
        if Ep/(1+2*Ep/me) > E:
            return 0 
        
        else:
            return jnp.pi* (aFS**2)/(me) * thermal_electron(T)/Ep**2 * (Ep/E + E/Ep +(me/E - me/Ep)**2 - 2 * me * (1/E - 1/Ep))
    '''
    @eqx.filter_jit 
    def inverse_compton_kernel_photon(self, E, T, Ep):
        """
        #does run faster with quadgk
        Eq B.20 in Hufnagel 2018

        K^IC_e-+->g
        """

        #if Ep < E + me:
        #    return 0
        
        ep_ll = E*me**2 /(4*Ep*(Ep-E))
        #ep_ul = min(E, Ephb_T_max*T, Ep-me**2 /(4*Ep))
        min1 = jnp.minimum(E, Ephb_T_max*T)
        ep_ul = jnp.minimum(Ep-me**2 /(4*Ep), min1)
        #if ep_ll >= ep_ul:
        #    return 0

        
        I_ic =  2 * jnp.pi * aFS**2 /Ep**2 * quadgk(ickp_integrand, [jnp.log(ep_ll), jnp.log(ep_ul)], epsrel= eps, epsabs= 0, args=(E, T, Ep))[0]
        return jnp.select([Ep < E + me, ep_ll >= ep_ul], [0, 0], default=I_ic) 
     
    @eqx.filter_jit 
    def total_kernel_photon(self, E, T, Ep, X: int):
        """Returns the total integration kernel for photon spectrium

        Parameters
        ----------
        E: float
            Energy of outgoing particle (MeV)
        T: float
            Temperature of backgroun photons (MeV)
        Ep: float
            Energy of incoming particle (MeV)
        X: int
            Which type of reaction 
                X = 0: photon to photon
                X = 1: electron to photon
                X = 2: positron to photon
        
        Returns
        -------
        Sum of photon integration kernels (Dimensionless)
        
        if X == 0:
            return self.photon_photon_scattering_kernel(E, T, Ep) + self.compton_scattering_kernel_photon(E, T, Ep) 
        elif X == 1:
            return self.inverse_compton_kernel_photon(E, T, Ep)
        elif X == 2:
            return self.inverse_compton_kernel_photon(E, T, Ep)
        else:
            raise ValueError("Invalid reaction type")
        """
        
        return jnp.select([X == 0, X == 1, X == 2], [self.photon_photon_scattering_kernel(E, T, Ep) + self.compton_scattering_kernel_photon(E, T, Ep), 
                                                     self.inverse_compton_kernel_photon(E, T, Ep), 
                                                     self.inverse_compton_kernel_photon(E, T, Ep)])
        
    #Note: e+ and e- does not have to be the same, but in this case, they are. 
    #This will be written to somewhat be general, so more electron/position reactions can be easily added

    ##############
    #Lepton Rates#
    ##############
    
    @eqx.filter_jit 
    def inverse_compton_rate(self, E, T):
        """ 
        Inverse Compton Scattering Rate Eq. B.24 in Hufnagel 2018
        
        Parameters
        ----------
        E: float
            Energy of outgoing particle (MeV)
        T: float
            Temperature of the thermal bath (MeV)

        Returns
        -------
        float
            Gamma^IC_e+- (MeV)
            
        """

        @jax.jit
        def inside_int(E_ph, E, ep):
            return F_func(E_ph, E, ep)
        
        @jax.jit
        def integral_of_inside(ep, E):
            return quadgk(inside_int, [ep, 4*ep*E**2 /(me**2 + 4*ep*E)], epsrel = eps, epsabs=0, args = (E, ep))[0]

        @jax.jit
        def outside_integral(ep, T, E):
            return integral_of_inside(ep, E) * f_thermal_photon_spec(ep, T)/ep
        
        ep_ll = 0
        ep_ul = jnp.minimum(E - (me**2)/(4*E), Ephb_T_max*T)
        #E_ph_ll = lambda x: x
        #E_ph_ul = lambda x: 4*x*E**2 /(me**2 + 4*x*E)

        #return 2*jnp.pi*aFS**2 /E**2 * dblquad(icre_integrand, ep_ll, ep_ul, E_ph_ll, E_ph_ul,  epsrel=eps, epsabs=0, args=(E, T))[0]
        return 2*jnp.pi*aFS**2 /E**2 * quadgk(outside_integral, [ep_ll, ep_ul], epsrel=eps, epsabs=0, args=(T, E))[0]
    
    @eqx.filter_jit 
    def total_lepton_rate(self, E, T):
        """
        Returns the total interaction rate for leptons (MeV)
        """

        return self.inverse_compton_rate(E, T)
    
    ################
    #Lepton Kernels#
    ################

    #has kernels for end state being leptons

    @eqx.filter_jit 
    def dphoton_pair_prod_kernel(self, E, T, Ep):
        """Eq B.6 in Hufnagel 2018

        Parameters:
        E:  (float)Energy of outgoing particle (MeV)
        T (float): Temperature (MeV)
        Ep(float): Energy of incoming particle (MeV)

        Returns
        -------
        K^DP_e+- (dimensionless)
        """

        #same limits as rate function
        '''
        if Ep < me**2 /(50*T):
            return 0
        '''
        dE = Ep - E
        ep_ll = jnp.maximum(Ep*( me**2 - 2.*dE*( jnp.sqrt(E**2 - me**2) - E ) )/( 4*Ep*dE + me**2 ), me**2 /Ep)
        min1 = jnp.minimum(Ep*( me**2 + 2.*dE*( jnp.sqrt(E**2 - me**2) + E ) )/( 4*Ep*dE + me**2 ), Ep)
        ep_ul = jnp.minimum(min1, Ephb_T_max*T) 

        #if ep_ll >= ep_ul:
        #    return 0
    

        dpk = jnp.pi * aFS**2 * me**2 /(4*Ep**3)* quadgk(dfk_integrand, [jnp.log(ep_ll), jnp.log(ep_ul)], args=(E, Ep, T))[0]

        return jnp.select([Ep < me**2 /(50*T), ep_ll >= ep_ul], [0, 0], default=dpk)
        #return jnp.pi * aFS**2 * me**2 /(4*Ep**3)* quad(dfk_integrand, jnp.log(ep_ll), jnp.log(ep_ul), args=(E, Ep, T))[0]

    @eqx.filter_jit 
    def bethe_heitler_pair_prod_kernel(self, E, T, Ep):
        """Bethe Heitler pair production kernel (g -> e+-) Equations B13-16 in Hufnagel 2018
        

        Parameters
        ----------
        E : float
            Energy of outgoing particle (MeV)
        T : float
            Temperature(MeV)
        Ep : float
            Energy of incoming particle (MeV)

        Returns
        -------
        float
            K^BH_e+- (dimensionless)
        """

        #if Ep - E - me < 0:
        #    return 0

        Eminus = E
        Eplus = Ep - E
        pp = jnp.sqrt(Eplus**2 - me**2)
        pm = jnp.sqrt(Eminus**2 - me**2)
        L = jnp.log((Eplus*Eminus + pp*pm + me**2)/(Eplus*Eminus - pp*pm + me**2))
        lp = jnp.log((Eplus + pp)/(Eplus - pp))
        lm = jnp.log((Eminus + pm)/(Eminus - pm))

        prefac  = aFS**3 / me**2 *(pp*pm/(Ep**3))
        dsigmadE =  -4/3 - 2 * Eplus * Eminus * (pp**2 + pm**2)/(pp**2 * pm**2) \
                    + me**2 * (lm*Eplus/(pm**3) + lp*Eminus/(pp**3) - lp*lm/(pp*pm)) \
                    + L * (-8*Eplus*Eminus/(3*pp*pm) + (Ep**2)/(pp**3 * pm**3)*(Eplus**2 * Eminus**2 + pp**2 * pm**2 - me**2 * Eplus * Eminus)) \
                    - L* me**2 * Ep /(2*pp*pm) *(lp * (Eplus * Eminus - pp**2)/(pp**3) + lm * (Eminus * Eplus - pm**2)/(pm**3))
        
        value = Z2n(T) * prefac * dsigmadE
        return jnp.select([Ep - E - me >= 0], [value], default=0)

    @eqx.filter_jit 
    def compton_scattering_kernel_electron(self, E, T, Ep):
        #in paper it says only electron but I dont see why it wouldnt also happen with positrons
        #Code has it for both so probably a typo
        return self.compton_scattering_kernel_photon(Ep + me - E, T, Ep)

    @eqx.filter_jit
    def inverse_compton_kernel_lepton(self, E, T, Ep):

        #definition of q (Bose Einstein distro would diverge)
        #if Ep == E:
        #    return 0
            
        #quadratic equation for limits
        a = 2 * Ep
        b = -4 * Ep * E - me**2
        c = (Ep - E)* me**2

        ep_ll = (-b - jnp.sqrt(b**2 - 4*a*c))/(2*a)
        min1 = jnp.minimum((-b + jnp.sqrt(b**2 - 4*a*c))/(2*a), (4*Ep - me**2)/(4*Ep))
        ep_ul = jnp.minimum(min1, Ephb_T_max*T)

        #if ep_ul < ep_ll:
        #    return 0
        
        Iick = 2 * jnp.pi * aFS**2 /Ep**2 * quadgk(icke_integrand, [jnp.log(ep_ll), jnp.log(ep_ul)], args=(E, T, Ep))[0]

        return jnp.select([Ep == E, ep_ll > ep_ul], [0, 0], default=Iick)
    
    @eqx.filter_jit
    def total_kernel_electron(self, E, T, Ep, X: int):
        """Returns the total integration for photon spectrium

        Parameters
        ----------
        E (float): Energy of outgoing particle (MeV)
        T (float): Temperature of backgroun photons (MeV)
        Ep(float): Energy of incoming particle (MeV)
        X (int): which type of reaction
                X = 0: photon to electron
                X = 1: electron to electron
                X = 2: positron to electron
        
        Returns
        -------
        Sum of electron integration kernels (Dimensionless)
        

        if X == 0:
            return self.dphoton_pair_prod_kernel(E, T, Ep) + self.compton_scattering_kernel_electron(E, T, Ep) + self.bethe_heitler_pair_prod_kernel(E, T, Ep)
        elif X == 1:
            return self.inverse_compton_kernel_lepton(E, T, Ep)
        elif X == 2:
            return 0
        else:
            raise ValueError("Invalid reaction type")
        """

        return jnp.select([X == 0, X == 1, X == 2], [self.dphoton_pair_prod_kernel(E, T, Ep) + self.compton_scattering_kernel_electron(E, T, Ep) + self.bethe_heitler_pair_prod_kernel(E, T, Ep),
                                                    self.inverse_compton_kernel_lepton(E, T, Ep),
                                                        0])
        
    @eqx.filter_jit
    def total_kernel_positron(self, E, T, Ep, X: int):
        """Returns the total integration for positron spectrium

        Parameters
        ----------
        E: float 
            Energy of outgoing particle (MeV)
        T: float 
            Temperature of backgroun photons (MeV)
        Ep: float 
            Energy of incoming particle (MeV)
        X: int
            which type of reaction
                X = 0: photon to positron
                X = 1: electron to positron
                X = 2: positron to positron

        Returns
        -------
        Sum of positron integration kernels (Dimensionless)
        

        if X == 0:
            return self.dphoton_pair_prod_kernel(E, T, Ep) + self.compton_scattering_kernel_electron(E, T, Ep) + self.bethe_heitler_pair_prod_kernel(E, T, Ep)
        elif X == 1:
            return 0
        elif X == 2:
            return self.inverse_compton_kernel_lepton(E, T, Ep)
        else:
            raise ValueError("Invalid reaction type")
        """

        return jnp.select([X == 0, X == 1, X == 2], [self.dphoton_pair_prod_kernel(E, T, Ep) + self.compton_scattering_kernel_electron(E, T, Ep) + self.bethe_heitler_pair_prod_kernel(E, T, Ep), 
                                                     0, 
                                                     self.inverse_compton_kernel_lepton(E, T, Ep)])
        
    @eqx.filter_jit
    def rate_x(self, X: int, E, T):
        """Returns rate for species type

        Parameters
        ----------
        X : int
            Species type
                X = 0: photon
                X = 1: Electron
                X = 2: Positron
        E : float
            Energy(MeV)
        T : float
            Temperature(MeV)

        Returns
        -------
        float
            returns total rate (MeV)
        
        if X == 0:
            return self.total_rate_photon(E, T)
        if X == 1:
            return self.total_lepton_rate(E, T)
        if X == 2:
            return self.total_lepton_rate(E, T)
        else:
            raise ValueError("Invalid reaction type")
        """
        
        return jnp.select([X == 0, X == 1, X == 2], [self.total_rate_photon(E, T), self.total_lepton_rate(E, T), self.total_lepton_rate(E, T)])
        
    @eqx.filter_jit
    def kernel_x(self, X:int, X_out: int, E, T, Ep):
        """Returns kernel for species type

        Parameters
        ----------
        X : int
            Species type
                X = 0: photon
                X = 1: Electron
                X = 2: Positron
        E : float
            Energy of outgoing particle(MeV)
        T : float
            Temperature(MeV)
        Ep : float
            Energy of inncoming particle (MeV)

        Returns
        -------
        float
            returns total kernel (dimensionless)
        
        if X == 0:
            return self.total_kernel_photon(E, T, Ep, X_out)
        if X == 1:
            return self.total_kernel_electron(E, T, Ep, X_out)
        if X == 2:
            return self.total_kernel_positron(E, T, Ep, X_out)
        else:
            raise ValueError("Invalid reaction type")
        """

        return jnp.select([X == 0, X == 1, X == 2], [self.total_kernel_photon(E, T, Ep, X_out), self.total_kernel_electron(E, T, Ep, X_out), self.total_kernel_positron(E, T, Ep, X_out)])  
    
    #@eqx.filter_jit
    def get_spectrum(self, E0, S_0f, S_contf, T):
        """Generate on-thermal spectrum of injected particles

        Parameters
        ----------
        E0 : float
            Injected energy (MeV)
        S_0 : array
            Monochromatic source term (MeV^4). This is MeV^4 since it gets multiplied by a delta function
        S_cont : array
            Continuous source term (MeV^3)
        T : float
            Temperature

        Returns
        ---------
        array
        first column is temperature (MeV)
        Other columns are the spectrum for each species (each X)
        """


        #number of grid points per decade energy so it does not fall below the min
        NE = jnp.array(jnp.log10(E0/Emin)*NE_pd, int)
        NE = jnp.maximum(NE, NE_min)
        NE = 5

        #number of species
        N_X = 3

        E_grid = jnp.logspace(jnp.log(Emin), jnp.log(E0), NE, base=jnp.e)

        # Generate the grid for the different species
        X_grid = jnp.arange(N_X)

        #rate (MeV)
        t1 = time.time()
        #ii, jj = jnp.meshgrid(X_grid, E_grid, indexing='ij')
        #R = jnp.reshape(jax.vmap(self.rate_x, in_axes=(0, 0, None))(ii.flatten(), jj.flatten(), T), (N_X, NE))
        R = jnp.array([[self.rate_x(X, E, T) for E in E_grid] for X in X_grid])
        #R = jnp.array([self.rate_x(X_grid, E, T) for E in E_grid])

        t2 = time.time()
        print("Rate time: ", t2-t1)

        ii, jj, kk, ll = jnp.meshgrid( X_grid, X_grid,E_grid, E_grid,indexing='ij')
        k = jnp.stack([ll, kk, jj, ii], axis=-1)
        k_true = jnp.array(k[:, :, :, :, 0] >= k[:, :, :, :, 1])
        K = jnp.select([k_true], [jnp.reshape(jax.vmap(self.kernel_x, in_axes=(0, 0, 0, None, 0))(ii.flatten(), jj.flatten(), kk.flatten(), 0.01, ll.flatten()), (N_X, N_X, NE, NE))])
        t3 = time.time()
        
        print("Kernel time: ", t3-t2)
        #K = jnp.select([k_true], [jnp.array([[[[self.kernel_x(X, X_out, E, T, Ep) for Ep in E_grid] for E in E_grid] for X_out in X_grid] for X in X_grid])])
        #K = jnp.array([[[[self.kernel_x(X, X_out, E, T, Ep) if Ep >= E else 0. for Ep in E_grid] for E in E_grid] for X_out in X_grid] for X in X_grid])

        S_0 = jnp.array([S0X(T) for S0X in S_0f])
        S_cont = jnp.array([[SCX(E, T) for E in E_grid] for SCX in S_contf])
        t4 = time.time()
        print("Source time: ", t4-t3)
        sol = solve_cascade_equation(E_grid, R, K, S_0, S_cont, T)
        t5 = time.time()
        print("Solve time: ", t5-t4)
        return sol


########################
#JIT Compiled Functions#
#######################


#General functions
@jax.jit
def f_thermal_photon_spec(ep, T):
    """Calculates the thermal photon spectrum ie the photon bath

    Parameters
    ----------
    ep: float
        Energy which to integrate over (MeV)
    T: float
        temperature (MeV)

    Returns
    -------
    float
        thermal photon spectrum (MeV^2)
    """

    return (ep**2)/jnp.pi**2 * 1/(jnp.exp(ep/T) - 1)

@jax.jit
def baryon_density(T):
    """
    Baryon to photon ratio in MeV^3.

    Parameters
    ----------
    T : float
        Temperature of the thermal bath in MeV.

    Returns
    -------
    Float
        Baryon density (eta) in MeV^3
    """

    return eta0 * 2 * zeta_3 / jnp.pi**2 * T**3

@jax.jit
def thermal_electron(T):
    """
    Thermal electron density relative to photons in MeV^3 calculated via charge neutrality
        ie n_e = Z*n_N where Z is the atomic number of element N
        Only H and He4 are considered in this case since they are much more abundant than the others
        Note: This neglects the temperature dependence of nuclear abundances and assumes Y_0 ie after BBN.

    Parameters
    ----------
    T: float
        Temperature in MeV.

    Returns
    -------
    n_e: float
        Thermal electron density relative to photons in MeV^3.

    """
    return (acro_Y["p"] + 2*acro_Y["He4"])*baryon_density(T)

@jax.jit
def Z2n(T):
    """
    Sum of (Z_N)^2 * n_N for Bethe Heitler pair creation

    Parameters
    ----------
    T: float
        Temperature of the thermal bath in MeV

    Returns
    -------
    float
        (Z_N)^2 * n_N in MeV^3

    """
    return (acro_Y["p"] + 4*acro_Y["He4"])*baryon_density(T)

@jax.jit
def F_func(E, Ep, ep):   
        """Function used for calculating rates and kernels related to inverse compton scattering 

        Parameters
        ----------
        E: float 
            Energy of outgoing particle (MeV)
        Ep: float 
            E', energy of incoming particle (MeV)
        ep: float
            epsilon, energy of thermal bath

        Returns
        -------
        float 
            F(E, E', epsilon), which is (dimensionless)
        """
        
        
        #should never happen if limits are right
        '''
        if not (ep <= E <= (4*ep*Ep**2)/(me**2 + 4*ep*Ep)):
            return 0
        '''
        
        G_ep = 4*ep*Ep/(me**2)
        q = E/(G_ep*(Ep-E))
        F = 2*q*jnp.log(q) + (1+2*q)*(1-q) + G_ep**2 * q**2 * (1 - q)/(2+2*G_ep*q)

        return F

#Double Photon Pair Production
@jax.jit
def dpr_integrand(s, ep, T):
    #Eq B.6 Hufnagel 2018 double photon rate integrand
    s = jnp.exp(s)
    ep = jnp.exp(ep)

    b = jnp.sqrt(1 - (4*me**2)/s)
    dp_cross_section = jnp.pi * aFS**2 /(2*me**2) * (1-b**2) * ((3-b**4) * jnp.log((1+b)/(1-b)) - 2*b *(2-b**2))

    return f_thermal_photon_spec(ep, T)/ep**2 * dp_cross_section * s*s*ep


@jax.jit
def dfk_integrand(ep, E, Ep, T):
    #Eq B.7 Hufnagel 2018
    ep = jnp.exp(ep)

    #check limits for valid G but shouldnt happen
    '''
    E_lim_plus = 0.5*(Ep + ep + (Ep - ep) * jnp.sqrt(1-me**2 /(Ep * ep)))
    E_lim_minus = 0.5*(Ep + ep - (Ep - ep) * jnp.sqrt(1-me**2 /(Ep * ep)))
    if not (me < E_lim_minus <= E <= E_lim_plus):
        return 0
    '''

    term1 = 4*(Ep + ep)**2 /(E*(Ep + ep - E)) * jnp.log(4 * ep * E * (Ep + ep - E)/(me**2 * (Ep + ep)))
    term2 = (me**2 /(ep*(Ep + ep)) - 1) * (Ep + ep)**4 / (E**2 * (Ep + ep - E)**2)
    term3 = 2 * (2*ep*(Ep + ep) -me**2) * (Ep + ep)**2 /(me**2 * E * (Ep + ep - E)) - 8* ep *(Ep + ep)/me**2
    G = term1 + term2 + term3

    return f_thermal_photon_spec(ep, T)/ep**2 * G * ep

#Inverse Compton Scattering
@jax.jit
def ickp_integrand(ep, E, T, Ep):
    ep = jnp.exp(ep)
    #should be taken care of by limit construction
    '''         
    if (ep >= E) or (E >= E*ep*Ep**2 /(me**2 + 4*ep*Ep)):
        return 0
    else:
    '''
    F = F_func(E, Ep, ep)
    return f_thermal_photon_spec(ep, T)/ep * F * ep
    
@jax.jit
def icre_integrand(E_ph, ep, E, T):
    #equation B.23 in Hufnagel 2018
    F = F_func(E_ph, E, ep)
    return f_thermal_photon_spec(ep, T)/ep * F

@jax.jit
def icke_integrand(ep, E, T, Ep):
    ep = jnp.exp(ep)
    F = F_func(Ep + ep - E, Ep, ep)
    return f_thermal_photon_spec(ep, T)/ep * F * ep

@jax.jit
def set_spectra(F, i, Fi, cond = False):
    """Manually sets photon spectrium to 0 in compressed regions so no floating point errors

    Parameters
    ----------
    F : array
        spectrum with indices (X, i)
    i : int
        index
    Fi : array
        new values
    cond : bool, optional
        if true, set to 0 to avoid errors, by default False
    """
    #Instead of x[idx] = y, use x = x.at[idx].set(y)
    #F[:, i] = Fi
    
    if cond:
        #F[0, i] = 0
        return F.at[0, i].set(0)
    else:
        return F.at[:, i].set(Fi)

@jax.jit
def solve_cascade_equation(E_grid, R, K, S0, SC, T):
    """
    Solves the cascade equation for the injected spectrum.
    The equation is a coupled Volterra equation of type 2 and the integrals are
    solved via trapezoial rule. 

    Parameters
    ----------
    E_grid : array
        Energy grid (MeV)
    R : array
        Rate (MeV)
    K : array
        Kernel (dimensionless)
    S_0 : array
        Monochromatic source term (MeV^4)
    S_cont : array
        Continuous source term (MeV^3)
    T : float
        Temperature (MeV)

    Returns
    -------
    array
        first column is temperature (MeV)
        Other columns are the spectrum for each species (each X)
    """

    N_X = len(R)
    NE = len(E_grid)

    dy = jnp.log(E_grid[-1]/Emin)/(NE-1)

    #create the grid to store the spectrums
    F_grid = jnp.zeros((N_X, NE)) 

    #calculate the last row, which is important for trapezoidal rule
    FX_E0 = jnp.array([SC[X,-1]/R[X,-1] + jnp.sum(K[X,:,-1,-1]*S0[:]/(R[:,-1]*R[X,-1])) for X in range(N_X)])

    F_grid = set_spectra(F_grid, -1, FX_E0)
    i  = (NE - 1) - 1 # start at the second to last index, NE-2

    #def cond_func(i):
    #    return i >= 0
    
    #def body_func(i):
    while i >= 0: # Counting down
        B = jnp.zeros( (N_X, N_X) )
        a = jnp.zeros( (N_X,   ) )

        I = jnp.identity(N_X)
        # Calculate the matrix B and the vector a
        for X in range(N_X):
            # Calculate B, : <--> Xp
            B = B.at[X,:].set(-.5*dy*E_grid[i]*K[X,:,i,i] + R[X,i]*I[X,:])

            # Calculate a
            a = a.at[X].set(SC[X,i])
            for Xp in range(N_X):
                a = a.at[X].add(K[X,Xp,i,-1]*S0[Xp]/R[Xp,-1] + .5*dy*E_grid[-1]*K[X,Xp,i,-1]*F_grid[Xp,-1])
                for j in range(i+1, NE-1): # Goes from i+1 to NE-2
                    a = a.at[X].add(dy*E_grid[j]*K[X,Xp,i,j]*F_grid[Xp,j])

        # Solve the system of linear equations of the form BF = a
        F_grid = set_spectra(F_grid, i,
            jnp.linalg.solve(B, a)
        )

        i -= 1

    #F_grid = jax.lax.while_loop(cond_func, body_func, (F_grid, (NE - 1) - 1))

    approx_zero = 1e-200
    # Remove potential zeros
    F_grid = F_grid.reshape( N_X*NE )
    F_grid = F_grid.at[:].set(jnp.where(F_grid > approx_zero, F_grid, approx_zero))
    #for i, f in enumerate(F_grid):
        #if f < approx_zero:
        #    F_grid[i] = approx_zero
    #    F_grid.at[i].set(f < approx_zero, f, approx_zero)
    F_grid = F_grid.reshape( (N_X, NE) )

    # Define the output array...
    sol = jnp.zeros( (N_X+1, NE) )
    # ...and fill it
    sol = sol.at[0     , :].set(E_grid)
    sol = sol.at[1:N_X+1, :].set(F_grid)

    return sol
