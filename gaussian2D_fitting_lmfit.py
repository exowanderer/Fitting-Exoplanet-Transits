# Load a set of lmfit modules to compare the efficacy of each approach
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit, Model
import numpy as np

def gaussian2D(params, yy,xx):
    amp    = params['amp']
    off    = params['off']
    ycen   = params['ycen']
    ywid   = params['ywid']
    xcen   = params['xcen']
    xwid   = params['xwid']
    return amp * np.exp(-0.5*(((yy -ycen)/ywid)**2. + ((xx -xcen)/xwid)**2.)) + off

def gaussian2D_Model(amp, off, ycen, ywid, xcen, xwid, yy, xx):
    return amp * np.exp(-0.5*(((yy -ycen)/ywid)**2. + ((xx -xcen)/xwid)**2.)) + off

def gaussian2D_residuals(params, yy,xx, data):
    """ model decaying sine wave, subtract data"""
    model = gaussian2D(params, yy, xx)
    return (model - data).flatten()

# Create True Parameters
amp   = 10.
off   = 1.
ycen  = 19.
xcen  = 31.
ywid  = 5.
xwid  = 7.

paramsTrue = Parameters()
paramsTrue.add('amp' , value=amp )
paramsTrue.add('off' , value=off )
paramsTrue.add('ycen', value=ycen)
paramsTrue.add('ywid', value=ywid)
paramsTrue.add('xcen', value=xcen)
paramsTrue.add('xwid', value=xwid)

np.random.seed(42)
nPts       = 50
noiseLevel = 1

yy,xx = np.indices((nPts,nPts))
data  = np.random.normal(gaussian2D(paramsTrue, yy, xx), noiseLevel)
# data  = amp * np.exp(-0.5*(((yy -ycen)/ywid)**2. + ((xx -xcen)/xwid)**2.))

# create a set of Parameters
paramsInit = Parameters()
paramsInit.add('amp' , value= 1., min=0.0, max=100)
paramsInit.add('off' , value= 0.0, min=0.0, max=100)
paramsInit.add('ycen', value= 25., min=0.0, max=50 )
paramsInit.add('ywid', value= 1.0, min=0.0, max=25 )
paramsInit.add('xcen', value= 25., min=0.0, max=50 )
paramsInit.add('xwid', value= 1.0, min=0.0, max=25 )

# do fit, here with leastsq model
# minner = Minimizer(gaussian2D_residuals, paramsInit, fcn_args=(yy,xx,data))
# result = minner.minimize()

# # calculate final result
# final = data + result.residual.reshape(data.shape)

# # write error report
# report_fit(result)

# do fit, here with leastsq model
gmodel = Model(gaussian2D_Model, independent_vars=['yy', 'xx'])

fitResult = gmodel.fit(data   = data,
                       params = paramsInit,
                       yy     = yy,
                       xx     = xx,
                       # weights= 1/derr,
                       method = 'powell')

print(fitResult.best_values)

bestModel = gaussian2D(fitResult.best_values, yy, xx)

# try to plot results
from pylab import figure, subplot, imshow, show
figure(figsize=(10,20))
subplot(121)
imshow(data)
subplot(122)
imshow(bestModel)
show()
