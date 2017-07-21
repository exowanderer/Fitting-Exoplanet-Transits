# Load a set of lmfit modules to compare the efficacy of each approach
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit, Model
import numpy as np

def gaussian2D(params, yy,xx):
    amp   = params['amp']
    off   = params['off']
    ycen   = params['ycen']
    ywid   = params['ywid']
    xcen   = params['xcen']
    xwid   = params['xwid']
    return amp * np.exp(-0.5*(((yy -ycen)/ywid)**2. + ((xx -xcen)/xwid)**2.)) + off

def gaussian2D_residuals(params, yy,xx, data):
    """ model decaying sine wave, subtract data"""
    return model - data

# create data to be fitted
yy,xx = np.indices((50,50))
amp   = 10.
off   = 1.
ycen  = 19.
xcen  = 31.
ywid  = 5.
xwid  = 7.
data  = amp2D * np.exp(-0.5*(((yy -ycen)/ywid)**2. + ((xx -xcen)/xwid)**2.))

# create a set of Parameters
params = Parameters()
params.add('amp', value= 10,  min=0, max=100)
params.add('off', value= 0,  min=0, max=100)
params.add('ycen', value= 25., min=0,max=50)
params.add('ywid', value= 1.0, min=0.0, max=25)
params.add('xcen', value= 25., min=0,max=50)
params.add('xwid', value= 1.0, min=0.0, max=25)

# do fit, here with leastsq model
minner = Minimizer(fncMin, params, fcn_args=(x,data))
result = minner.minimize()

# calculate final result
final = data + result.residual

# write error report
report_fit(result)

# try to plot results
try:
    import pylab
    pylab.plot(x, data, 'k+')
    pylab.plot(x, final, 'r')
    pylab.show()
except:
    pass
