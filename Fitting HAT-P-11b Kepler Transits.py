h11Per       = 4.88782433
h11t0        = 2454957.812464 - 2454833.0
h11Inc       = 88.99
h11ApRs      = 14.64
h11RpRs      = 0.05856
h11Ecc       = 0.26493
h11Omega     = 360-162.149
h11u1        = 0.646
h11u2        = 0.048

def batman_wrapper_raw(period, tcenter, inc, aprs, rprs, ecc, omega, u1, u2, p0, p1, p2, 
                       times, ldtype='quadratic', transitType='primary'):

    # period, tcenter, inc, aprs, rprs, ecc, omega, u1, u2, p0, p1, p2 = params
    
    if p0 == 1.0 and p1 == 0.0 and p2 == 0.0:
        out_of_transit = 1.0
    else:
        out_of_transit = p0 + p1*(times - times.mean()) + p2*(times - times.mean())**2.
    
    bm_params           = batman.TransitParams() # object to store transit parameters

    bm_params.per       = period  # orbital period
    bm_params.t0        = tcenter # time of inferior conjunction
    bm_params.inc       = inc     # inclunaition in degrees
    bm_params.a         = aprs    # semi-major axis (in units of stellar radii)
    bm_params.rp        = rprs    # planet radius (in units of stellar radii)
    bm_params.ecc       = ecc     # eccentricity
    bm_params.w         = omega   # longitude of periastron (in degrees)
    bm_params.limb_dark = ldtype              # limb darkening model # NEED TO FIX THIS
    bm_params.u         = [u1, u2]                  # limb darkening coefficients # NEED TO FIX THIS

    m_eclipse = batman.TransitModel(bm_params, times, transittype=transitType)    # initializes model

    return m_eclipse.light_curve(bm_params) * out_of_transit

def loglikehood(params, uni_prior, times, flux, fluxerr, regularization=None):
    model = batman_wrapper_raw(params, times)
    chisq = ((flux - model)/fluxerr)**2.
    if regularization is None:
        return -0.5*chisq.sum() # + lambda*abs(params).sum() # + lambda*np.sqrt((params**2).sum())
    elif regularization == 'Ridge':
        return -0.5*chisq.sum() + lambda*np.sqrt((params**2).sum())
    elif regularization = 'LASSO':
        return -0.5*chisq.sum() + lambda*abs(params).sum()

def logPrior(params, uni_prior, times, flux, fluxerr):
    for kp, (lower, upper) in enumerate(uni_prior):
        if params[kp] < lower or params[k] > upper:
            return -np.inf
        return 0.0

def logPosterior(params, uni_prior, times, flux, fluxerr):
    logPriorNow = logPrior(params, uni_prior, times, flux, fluxerr)
    logLikeLNow = loglikehood(params, uni_prior, times, flux, fluxerr)
    return logLikeLNow + logPriorNow

def neg_logprobability(params, uni_prior, times, flux, fluxerr):
    return -2*logPosterior(params, uni_prior, times, flux, fluxerr)

periodIn    = h11Per
tcenterIn   = h11t0
incIn       = h11Inc
aprsIn      = h11ApRs
rprsIn      = h11RpRs
eccIn       = h11Ecc
omegaIn     = h11Omega
u1In        = h11u1
u2In        = h11u2
offset      = 1.0
slope       = 0.0
quadterm    = 0.0

# Initial Parameters
initParams = [periodIn, tcenterIn, incIn, aprsIn, rprsIn, eccIn, omegaIn, u1In, u2In]

# Frozen Prior
uniPrior = np.array([
            [periodIn,periodIn],
            [tcenterIn, tcenterIn],
            [incIn, incIn],
            [aprsIn, aprsIn],
            [rprsIn, rprsIn],
            [eccIn,eccIn],
            [omegaIn,omegaIn],
            [u1In,u1In],
            [u2In,u2In]
           ])

# Partial UnFrozen Prior
uniPrior = np.array([
            [periodIn,periodIn], # uniform volume for period (== 0)
            [tcenterIn-0.1, tcenterIn+0.1], # uniform volume for tcenter (== 0.2)
            [80., 90.], # uniform volume for inclination
            [10, 20], # uniform volume for ApRs
            [0.01, 0.1], # uniform volume for RpRs
            [eccIn,eccIn], # uniform volume for ecc
            [omegaIn,omegaIn], # uniform volume for omega
            [0.6,0.7], # uniform volume for u1
            [0.0,0.1] # uniform volume for u2
           ])

res = optmin(neg_logprobability, initParams, args=(uniPrior, timeSlice3, fluxSlice3, ferrSlice3))
print(res.x - initParams) # it seems that the prior is overachieving -- maybe set the wrong init conditions

## FIRST TRY at LMFIT

from lmfit import Parameters, Model
p = Parameters()

p.add('h11Per', value = 4.88782433, vary=False)
p.add('h11t0', value = 2454957.812464 - 2454833.0, vary=True)
p.add('h11Inc', value = 88.99, vary=True)
p.add('h11ApRs', value = 14.64, vary=True)
p.add('h11RpRs', value = 0.05856, vary=True)
p.add('h11Ecc', value = 0.26493, vary=False)
p.add('h11Omega', value = 360-162.149, vary=False)
p.add('h11u1', value = 0.646, vary=True)
p.add('h11u2', value = 0.048, vary=True)
p.add('intercept', value = 1.0, vary=True)
p.add('slope', value = 0.0, vary=True)
p.add('curvature', value = 0.0, vary=True)
'''
df = pd.read_csv('w18.csv')
t = df['time'].values
orbit = df['orbit'].values
scanD = df['scanD'].values
f0 = df['flux0'].values
fitLCModel(params, times, flux0, scanD, df['weights'].values)
'''
lc = Model(batman_wrapper_raw, independent_vars=['times', 'ldtype', 'transitType'])

fitResult = lc.fit(f0,
                   tExp=tExp,
                   times=times,
                   ldtype='quadratic',
                   transitType='primary',
                   params=p,
                   method='powell')
print(fitResult)
