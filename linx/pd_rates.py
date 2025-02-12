import jax
import jax.numpy as jnp 
import equinox as eqx
from numpy import loadtxt

from linx.const import hbar, kB, me, aFS, E_EC_max, eps, NE_pd, approx_zero, NT_pd
from linx.thermo import T_g, Hubble, rho_EM_std_v
from linx.special_funcs import zeta_3
from linx.injected_spec import InjectedSpectrum
from quadax import quadgk

E_th = {
    "dgnp" :  2.224573,
    "tgnd" :  6.257248,
    "tgnpn" :  8.481821,
    "He3gpd" :  5.493485,
    "He3gnpp" :  7.718058,
    "He4gpt" : 19.813852,
    "He4gnHe3" : 20.577615,
    "He4gdd" : 23.846527,
    "He4gnpd" : 26.071100,
    "Li6gnpHe4" :  3.698892,
    "Li6gX" : 15.794685,
    "Li7gtHe4" :  2.467032,
    "Li7gnLi6" :  7.249962,
    "Li7gnnpHe4" : 10.948850,
    "Be7gHe3He4" :  1.586627,
    "Be7gpLi6" :  5.605794,
    "Be7gppnHe4" :  9.304680
}

#will probably have to add more classes depending on what models we want to look at

class pd_rxns(eqx.Module): #try to use this to subclass the other models 
    #mass: float
    #tau: float
    #n: float
    #filename: str
    pd_tables: jnp.array
    temp: jnp.array
    
    """
    Class for getting the reaction rates for decay model. This will be changed into
    more general models later.

    Attributes
    ----------
    mass : float
        Mass of particle in MeV
    tau : float
        Lifetime of particle in seconds
    n : float
        Number of particles normalized to number of photon (n_X/n_gamma)
    """

    #These are the interp functions for each reaction
    #def __init__(self, mass, tau, n):
    def __init__(self, pdi_grid):
        #self.mass = mass
        #self.tau = tau
        #self.n = n 
        #self.filename = f"../../acropolis/tables/{self.mass}_{self.tau}_{self.n}"
        #self.pd_tables = jnp.array(loadtxt(f"{self.filename}"))
        #self.pd_tables = pdi_grid
        #self.temp = pdi_grid[0]/kB
        self.temp = pdi_grid[0]/kB
        self.pd_tables = pdi_grid
        
    #d + g -> n + p
    def dgnp_frwrd_rate(self, T, p):
        temp = self.temp #change from MeV to Kelvin
        rate = self.pd_tables[1]/hbar
        #return jnp.interp(T, temp, rate, left=0, right=0)
        return jnp.interp(T, temp, rate, left=rate[0], right=rate[-1])

    #t + g -> n + d
    def tgnd_frwrd_rate(self, T, p):
        temp = self.temp
        rate = self.pd_tables[2]/hbar
        return jnp.interp(T, temp, rate, left=rate[0], right=rate[-1])

    #t + g -> n + p + n
    def tgnpn_frwrd_rate(self, T, p):
        temp = self.temp
        rate = self.pd_tables[3]/hbar
        return jnp.interp(T, temp, rate, left=rate[0], right=rate[-1])

    #He3 + g -> p + d
    def He3gpd_frwrd_rate(self, T, p):
        temp = self.temp
        rate = self.pd_tables[4]/hbar
        return jnp.interp(T, temp, rate, left=rate[0], right=rate[-1])

    #He3 + g -> n + p + p
    def He3gnpp_frwrd_rate(self, T, p):
        temp = self.temp
        rate = self.pd_tables[5]/hbar
        return jnp.interp(T, temp, rate, left=rate[0], right=rate[-1])

    #He4 + g -> p + t
    def He4gpt_frwrd_rate(self, T, p):
        temp = self.temp
        rate = self.pd_tables[6]/hbar
        return jnp.interp(T, temp, rate, left=rate[0], right=rate[-1])

    #He4 + g -> n + He3
    def He4gnHe3_frwrd_rate(self, T, p):
        temp = self.temp
        rate = self.pd_tables[7]/hbar
        return jnp.interp(T, temp, rate, left=rate[0], right=rate[-1]) 

    #He4 + g -> d + d
    def He4gdd_frwrd_rate(self, T, p):
        temp = self.temp
        rate = self.pd_tables[8]/hbar
        return jnp.interp(T, temp, rate, left=rate[0], right=rate[-1])

    #He4 + g -> n + p + d
    def He4gnpd_frwrd_rate(self, T, p):
        temp = self.temp
        rate = self.pd_tables[9]/hbar
        return jnp.interp(T, temp, rate, left=rate[0], right=rate[-1])
    #Li6 + g -> n + p + He4
    def Li6gnpHe4_frwrd_rate(self, T, p):
        temp = self.temp
        rate = self.pd_tables[10]/hbar
        return jnp.interp(T, temp, rate, left=rate[0], right=rate[-1])

    #Li6 + g -> X these are to Li5 which almost immediately decay
    def Li6gX_frwrd_rate(self, T, p):
        temp = self.temp
        rate = self.pd_tables[11]/hbar
        return jnp.interp(T, temp, rate, left=rate[0], right=rate[-1])

    #Li7 + g -> t + He4
    def Li7gtHe4_frwrd_rate(self, T, p):  
        temp = self.temp
        rate = self.pd_tables[12]/hbar
        return jnp.interp(T, temp, rate, left=rate[0], right=rate[-1])

    #Li7 + g -> n + Li6
    def Li7gnLi6_frwrd_rate(self, T, p):  
        temp = self.temp
        rate = self.pd_tables[13]/hbar
        return jnp.interp(T, temp, rate, left=rate[0], right=rate[-1])

    #Li7 + g -> n + n + p + He4
    def Li7gnnpHe4_frwrd_rate(self, T, p):  
        temp = self.temp
        rate = self.pd_tables[14]/hbar
        return jnp.interp(T, temp, rate, left=rate[0], right=rate[-1])

    #Be7 + g -> He3 + He4
    def Be7gHe3He4_frwrd_rate(self, T, p):    
        temp = self.temp
        rate = self.pd_tables[15]/hbar
        return jnp.interp(T, temp, rate, left=rate[0], right=rate[-1])

    #Be7 + g -> p + Li6
    def Be7gpLi6_frwrd_rate(self, T, p):    
        temp = self.temp
        rate = self.pd_tables[16]/hbar
        return jnp.interp(T, temp, rate, left=rate[0], right=rate[-1])

    #Be7 + g -> p + p + n + He4
    def Be7gppnHe4_frwrd_rate(self, T, p):    
        temp = self.temp
        rate = self.pd_tables[17]/hbar
        return jnp.interp(T, temp, rate, left=rate[0], right=rate[-1])
    
    #Could make this a new class? not sure
    """
    class pd_rxns(eqx.Module):

        def __init__(self):
            pass 
    """

    #Cross section calculations for reactions
    def generic_expr(self, E, Q, N, p1, p2, p3):
        """Many calculations for cross section follow this form. Look in 
        Updated Nucleosynthesis Constraints on Unstable Relic Particles (Cyburt et. al 2002)
        https://arxiv.org/pdf/astro-ph/0211258

        Parameters
        ----------
        E : float
            Photon Energy
        Q : float
            threshold energy
        N : float
            parameters
        p1 : float
            parameters
        p2 : float
            parameters
        p3 : float
            parameters

        Returns
        -------
        float
            cross section
        """

        return jnp.select([E > Q], [N * (Q**p1) * (E-Q)**p2 /(E**p3)], default=0)

    #d + g -> n + p
    def dgnp_cross(self, E):
        Q = E_th["dgnp"]
        return jnp.select([E > Q], [18.75 * ( (jnp.sqrt( Q*(E-Q) )/E )**3. + 0.007947*(jnp.sqrt( Q*(E-Q) )/E )**2. * ( (jnp.sqrt(Q) - jnp.sqrt(0.037))**2./( E - Q + 0.037 ) ) )])
    
    #t + g -> n + d
    def tgnd_cross(self, E):
        return self.generic_expr(E, E_th["tgnd"], 9.8, 1.95, 1.65, 3.6)
    
    #t + g -> n + p + n
    def tgnpn_cross(self, E):
        return self.generic_expr(E, E_th["tgnpn"], 26.0, 2.6, 2.3, 4.9)
    
    #He3 + g -> p + d
    def He3gpd_cross(self, E):
        return self.generic_expr(E, E_th["He3gpd"], 8.88, 1.75, 1.65, 3.4)
    
    #He3 + g -> n + p + p
    def He3gnpp_cross(self, E):
        return self.generic_expr(E, E_th["He3gnpp"], 16.7, 1.95, 2.3, 4.25)
    
    #He4 + g -> p + t
    def He4gpt_cross(self, E):
        return self.generic_expr(E, E_th["He4gpt"], 19.5, 3.5, 1.0, 4.5)
    
    #He4 + g -> n + He3
    def He4gnHe3_cross(self, E):
        return self.generic_expr(E, E_th["He4gnHe3"], 20.7, 3.5, 1.0, 4.5)
    
    #He4 + g -> d + d
    def He4gdd_cross(self, E):
        return self.generic_expr(E, E_th["He4gdd"], 10.7, 10.2, 3.4, 13.6)
    
    #He4 + g -> n + p + d
    def He4gnpd_cross(self, E):
        return self.generic_expr(E, E_th["He4gnpd"], 21.7, 4.0, 3.0, 7.0)
    
    #Li6 + g -> n + p + He4
    def Li6gnpHe4_cross(self, E):
        return self.generic_expr(E, E_th["Li6gnpHe4"], 143.0, 2.3, 4.7, 7.0)
    
    #Li6 + g -> X these are to Li5 which almost immediately decay
    def Li6gX_cross(self, E):
        Q = E_th["Li6gX"]

        def exp_term(E, N, Eb, Ed):
            return N * jnp.exp( -(1./2.)*( (E - Eb)/Ed )**2. )

        return self.generic_expr(E, Q, 38.1, 3.0, 2.0, 5.0) * (exp_term(E, 3.7, 19.0, 3.5) + exp_term(E, 2.75, 30.0, 3.0) + exp_term(E, 2.2, 43.0, 5.0) )

    #Li7 + g -> t + He4
    def Li7gtHe4_cross(self, E):
        Q = E_th["Li7gtHe4"]

        Ecm = E - Q
        pEcm = 1. + 2.2875*(Ecm**2.) - 1.1798*(Ecm**3.) + 2.5279*(Ecm**4.)

        return jnp.select([E > Q], [0.105 * ( 2371./(E**2) ) * jnp.exp( -2.5954/jnp.sqrt(Ecm) ) * jnp.exp(-2.056*Ecm) * pEcm], default=0)
  
    #Li7 + g -> n + Li6
    def Li7gnLi6_cross(self, E):
        Q = E_th["Li7gnLi6"]
        return jnp.select([E > Q], [self.generic_expr(E, Q, 0.176, 1.51, 0.49, 2.0) + self.generic_expr(E, Q, 1205.0, 5.5, 5.0, 10.5) + 0.06/( 1. + ( (E - Q - 7.46)/0.188 )**2. )], default=0)

    #Li7 + g -> n + n + p + He4
    def Li7gnnpHe4_cross(self, E):  
        return self.generic_expr(E, E_th["Li7gnnpHe4"], 122.0, 4.0, 3.0, 7.0)
    
    #Be7 + g -> He3 + He4
    def Be7gHe3He4_cross(self, E):    
        Q = E_th["Be7gHe3He4"]

        Ecm = E - Q
        pEcm = 1. - 0.428*(Ecm**2.) + 0.534*(Ecm**3.) - 0.115*(Ecm**4.)
        return jnp.select([E < Q, pEcm < 0], [0, 0], default = 0.504 * ( 2371./(E**2.) ) * jnp.exp( -5.1909/jnp.sqrt(Ecm) ) * jnp.exp(-0.548*Ecm) * pEcm)

    #Be7 + g -> p + Li6
    def Be7gpLi6_cross(self, E):    
        Q = E_th["Be7gpLi6"]
        return self.generic_expr(E, Q, 32.6, 10.0, 2.0, 12.0) + self.generic_expr(E, Q, 2.27e6, 8.8335, 13.0, 21.8335)
    
    #Be7 + g -> p + p + n + He4
    def Be7gppnHe4_cross(self, E):    
        return self.generic_expr(E, E_th["Be7gppnHe4"], 133.0, 4.0, 3.0, 7.0)

    def get_cross_section(self, E):
        """Returns an array of cross sections

        Parameters
        ----------
        E : float
            Energy

        Returns
        -------
        array
            Array of all cross sections and converts to mb to MeV^-2
        """
        
        return jnp.real(jnp.array([self.dgnp_cross(E), 
                self.tgnd_cross(E), 
                self.tgnpn_cross(E), 
                self.He3gpd_cross(E), 
                self.He3gnpp_cross(E), 
                self.He4gpt_cross(E), 
                self.He4gnHe3_cross(E), 
                self.He4gdd_cross(E), 
                self.He4gnpd_cross(E), 
                self.Li6gnpHe4_cross(E), 
                self.Li6gX_cross(E), 
                self.Li7gtHe4_cross(E), 
                self.Li7gnLi6_cross(E), 
                self.Li7gnnpHe4_cross(E), 
                self.Be7gHe3He4_cross(E), 
                self.Be7gpLi6_cross(E), 
                self.Be7gppnHe4_cross(E)])) * 2.56819e-6 # conversion factor for mb to MeV^-2
    
class decay_model(pd_rxns):
    mass: float
    tau: float
    n: float
    bree: float
    brgg: float
    t_vec: jnp.array
    a_vec: jnp.array
    T_g_vec: jnp.array
    rho_g_vec: jnp.array
    inject_temp: float
    inject_time: float
    E0: float
    spec: InjectedSpectrum
    temp: jnp.array
    pd_tables: jnp.array

    def __init__(self, mass, tau, n_X, bree, brgg, t_vec, a_vec, rho_g_vec, inject_temp = 0):
        """Class for DM particle X decaying into SM particles

        Parameters
        ----------
        mass : float
            Mass of DM particle in MeV
        tau : float
            lifetime of DM particle in seconds
        n_X : float
            Number density of DM particle normalized to photon number density (dimensionless)
        bree : float
            Electron branching ratio
        brgg : float
            Photon branching ratio
        rho_g_vec : array
            Array of energy density values in MeV^4 from background model
        t_vec : array
            Array of time values in seconds from background model in s
        a_vec : array
            Array of scale factor values from background model
        inject_temp : float (optional)
            Temperature at which the decaying particles are injected. 
            If 0, it is assumed that the particles decay at T(tau)
        """
        self.mass = mass
        self.tau = tau
        self.n = n_X
        self.bree = bree
        self.brgg = brgg
        #thermal bath temperatue
        self.T_g_vec = T_g(rho_g_vec)
        self.t_vec = t_vec
        self.rho_g_vec = rho_g_vec
        #self.rho_nu_vec = rho_nu_vec
        #self.rho_NP_vec = rho_NP_vec
        self.a_vec = a_vec
        self.spec = InjectedSpectrum()
        if inject_temp == 0:
            self.inject_temp = self.get_temp(self.tau)
        else:
            self.inject_temp = inject_temp

        #injection time
        self.inject_time = self.get_time(self.inject_temp)

        #injection energy: mass/2
        self.E0 = mass/2
        
        self.pd_tables = self.get_pdi_grids()
        #self.pd_tables = jnp.zeros((3, 17))
        #self.temp = self.pd_tables[0]/kB

        super().__init__(self.pd_tables)

    @eqx.filter_jit
    def get_scale_factor(self, T):
        """Returns scale factor given temperature

        Parameters
        ----------
        T : float
            Temperature (MeV)

        Returns
        -------
        a : float
            scale factor (dimensionless)
        """

        return jnp.interp(T, jnp.flip(self.T_g_vec), jnp.flip(self.a_vec), left=self.a_vec[-1]/2, right=self.a_vec[0]*2)
    
    @eqx.filter_jit
    def get_time(self, T):
        """Returns time given temperature

        Parameters
        ----------
        T : float
            Temperature (MeV)

        Returns
        -------
        t : float
            time (seconds)
        """

        return jnp.interp(T, jnp.flip(self.T_g_vec), jnp.flip(self.t_vec), left=self.t_vec[-1]/2, right=self.t_vec[0]*2)
    
    @eqx.filter_jit
    def get_temp(self, t):
        """Returns temperature given time

        Parameters
        ----------
        t : float
            time (seconds)

        Returns
        -------
        T : float
            Temperature (MeV)
        """

        return jnp.interp(t, self.t_vec, self.T_g_vec, left=self.T_g_vec[0]/2, right=self.T_g_vec[-1]*2)
    
    @eqx.filter_jit
    def number_density(self, T):
        """
        Returns number density of DM particles at temperature T not relative to photons for exponentially decaying species
        Equation 4 of Poulin et al. 2015. 

        Parameters
        ----------
        T : float
            Temperature (MeV)

        Returns
        -------
        float
            Number density of DM particles (MeV^3)
        """
        #(1+z)
        redshift_fac = self.get_scale_factor(self.inject_temp)/self.get_scale_factor(T)

        delta_t = self.get_time(T) - self.inject_time
        #photon density
        n_photon = (2*zeta_3) * self.inject_temp**3 /jnp.pi**2

        return self.n * redshift_fac**3 * n_photon * jnp.exp(-delta_t/self.tau)
    
    @eqx.filter_jit
    def temperature_range(self): 
        """Returns the temperature range of the model

        Returns
        -------
        tuple
            Minimum and maximum temperature (MeV)
        """
        #number of degrees of freedom
        mag = 2

        decay_temp = self.get_temp(self.tau)
        Tmin = 10.**(jnp.log10(decay_temp) - 3.*mag/4.)
        Tmax = 10.**(jnp.log10(decay_temp) + 1.*mag/4.)

        return (Tmin, Tmax)

    @eqx.filter_jit
    def S_photon_0(self, T):
        """Monochromatic photon source term at temperature T. Equation 3.3 in Hufnagel 2018

        Parameters
        ----------
        T : float
            Temperature (MeV)

        Returns
        -------
        float
            Source term (MeV^4) without the delta function
        """

        return self.brgg * 2 * self.number_density(T) * hbar / self.tau 

    @eqx.filter_jit
    def S_electron_0(self, T):
        """Monochromatic electron source term at temperature T. Equation 3.3 in Hufnagel 2018

        Parameters
        ----------
        T : float
            Temperature (MeV)

        Returns
        -------
        float
            Source term (MeV^4) without the delta function
        """

        return self.bree * self.number_density(T) * hbar / self.tau
    
    @eqx.filter_jit
    def S_photon_cont(self, E, T):
        """Continuous photon source term, which describes the final state radiation of X -> e+e-
          Equation 3.4 in Hufnagel 2018

        Parameters
        ----------
        E : float
            Energy injected (MeV)
        T : float
            Temperature (MeV)

        Returns
        -------
        float
            Source term (MeV^3)
        """

        E0 = self.E0
        x = E/E0
        S_e = self.S_electron_0(T)

        nonzero = S_e/E0 * aFS/jnp.pi * (1 + (1-x)**2)/x * jnp.log(4*E0**2 * (1-x)/ me**2)

        return jnp.select([1 - me**2 /(4 * E0**2) - x < 0], [0], default=nonzero)

        #rewritten for jax
        '''
        #heaviside condition
        if 1 - me**2 /(4 * E0**2) - x < 0:
            return 0
        
        return S_e/E0 * aFS/jnp.pi * (1 + (1-x)**2)/x * jnp.log(4*E0**2 * (1-x)/ me**2)
        '''

    @eqx.filter_jit
    def S_electron_cont(self, E, T):
        return jnp.full_like(E, 1e-200)
    
    @eqx.filter_jit
    def get_source_0(self):
        #gets the list of all monochromatic sources
        return [self.S_photon_0, self.S_electron_0, self.S_electron_0]

    @eqx.filter_jit
    def get_source_cont(self):
        #gets the list of all monochromatic sources
        return [self.S_photon_cont, self.S_electron_cont, self.S_electron_cont]

    #@eqx.filter_jit
    def pdi_rates(self, T):
        """Return photon disintegration rates

        Parameters
        ----------
        T : float
            temperature

        Returns
        -------
        dict
            Dictionary of the reaction rate for a certain temperature
        """

        EC = me**2/(22 * T)
        Emax = jnp.minimum(self.E0, E_EC_max*EC)
        #pdi_rates = {key: approx_zero for key in E_th.keys()}
        Eth_list = jnp.array(list(E_th.values()))
        sp = self.spec.get_spectrum(self.E0, self.get_source_0(), self.get_source_cont(), T)

        rate_photon_E0 = self.spec.total_rate_photon(self.E0, T)


        @jax.jit
        def F_s(log_E, i):
            E = jnp.exp(log_E)
            #return jnp.interp(E, sp[0], sp[1]) * E * self.get_cross_section(E)[i]
            return jnp.exp(jnp.interp(jnp.log(E), jnp.log(sp[0]), jnp.log(sp[1]))) * E * self.get_cross_section(E)[i]

        I_dt = self.S_photon_0(T) * self.get_cross_section(self.E0)/rate_photon_E0

        #pdi_rates = I_dt
        #pdi_rates = jnp.select([Emax > Eth_list],[jax.vmap(quadgk, in_axes=(None,0, None, None, None, None ))(F_s, [jnp.log(Eth_list), jnp.full_like(Eth_list, jnp.log(Emax))], (), False, 0, eps)])
        #pdi_rates = jnp.select([Emax > Eth_list], [jax.vmap(quadgk, in_axes=(None,0, None, None, None, None ))(F_s, jnp.array([jnp.log(Eth_list), jnp.full_like(Eth_list, jnp.log(Emax))]), (), False, 0, eps)[0]], default = I_dt)
        pdi_rates = jax.vmap(quadgk, in_axes=(None,1, 0, None, None, None ))(F_s, jnp.array([jnp.log(Eth_list), jnp.full_like(Eth_list, jnp.log(Emax))]), (jnp.arange(17), ), False, 0, eps)
        #pdi_rates = jax.lax.fori_loop(0, 18, lambda i: quadgk(F_s, [jnp.log(Eth_list[i]), jnp.log(Emax)], epsrel=eps, epsabs=0, args=(jnp.arange(17), ))[0])
        pdi_rates = jnp.select([Emax > Eth_list], [pdi_rates[0] + I_dt], default = I_dt)
        pdi_rates = jnp.maximum(pdi_rates, approx_zero)

        """
        for i, rkey in enumerate(E_th.keys()):

            I_dt = self.S_photon_0(T) * self.get_cross_section(self.E0)[i]/rate_photon_E0
            pdi_rates[rkey] = I_dt
            
            if Emax > E_th[rkey]:
                log_Emin, log_Emax = jnp.log(E_th[rkey]), jnp.log(Emax)
                I_Fs = quadgk(F_s, [log_Emin, log_Emax], epsrel=eps, epsabs=0, args=(i, ))
                pdi_rates[rkey] += I_Fs[0]

            pdi_rates[rkey] = jnp.maximum(pdi_rates[rkey], approx_zero)
        """

        return pdi_rates
    
    #@eqx.filter_jit
    def get_pdi_grids(self):
        """Return photon disintegration rates on a grid of temperatures
        
        Returns
        -------
        array
            Array of photon disintegration rates with sol[0] as the temperature grid
        """
        (Tmin, Tmax) = self.temperature_range()
        NT = int(jnp.log10(Tmax/Tmin)*NT_pd)

        #logspaced temperature grid
        Tr = jnp.logspace(jnp.log10(Tmin), jnp.log10(Tmax), 50)

        #rates = []
        #grid = Tr
        #for Ti in Tr:
        #    rates_at_i = self.pdi_rates(Ti)
        grid = jax.vmap(self.pdi_rates)(Tr)
            #rates.append(jnp.array(list(rates_at_i.values())))
            #rates.append(rates_at_i)
        #    grid = jnp.vstack(grid, rates_at_i)
        #rates = jnp.transpose(jnp.array(rates))

        return jnp.vstack([Tr, jnp.transpose(grid)])
        #return grid
    
        
