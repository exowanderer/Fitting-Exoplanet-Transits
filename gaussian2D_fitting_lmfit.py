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

# Create True Parameters
amp   = 10.
off   = 1.
ycen  = 19.
xcen  = 31.
ywid  = 5.
xwid  = 7.

paramsTrue = Parameters()
paramsTrue.add('amp' , value=amp)
paramsTrue.add('off' , value=off)
paramsTrue.add('ycen', value=ycen)
paramsTrue.add('ywid', value=ywid)
paramsTrue.add('xcen', value=xcen)
paramsTrue.add('xwid', value=xwid)

nPts  = 50
yy,xx = np.indices((nPts,nPts))
data  = gaussian2D(paramsTrue, yy, xx)
# data  = amp * np.exp(-0.5*(((yy -ycen)/ywid)**2. + ((xx -xcen)/xwid)**2.))

# create a set of Parameters
paramsInit = Parameters()
paramsInit.add('amp' , value= 10., min=0.0, max=100)
paramsInit.add('off' , value= 0,0, min=0.0, max=100)
paramsInit.add('ycen', value= 25., min=0.0, max=50 )
paramsInit.add('ywid', value= 1.0, min=0.0, max=25 )
paramsInit.add('xcen', value= 25., min=0.0, max=50 )
paramsInit.add('xwid', value= 1.0, min=0.0, max=25 )

# do fit, here with leastsq model
minner = Minimizer(gaussian2D_residuals, paramsInit, fcn_args=(yy,xx,data))
result = minner.minimize()

# calculate final result
final = data + result.residual.reshape(data.shape)

# write error report
report_fit(result)

bestModel = gaussian2D(results.params, yy, xx)

# try to plot results
figure(figsize=(10,20))
subplot(211)
imshow(data)
subplot(121)
imshow(bestModel)
