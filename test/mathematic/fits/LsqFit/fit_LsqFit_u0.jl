"""
  When `û → 0`
"""

modelDM(v,p) = p[1] * modelDMexp(v,p[2:3],L1-1)

uu = ûa[isp3]
p=[1.0, 1.0, uu]
lbs = [-20.0, 0.0, 0]
ubs = [20.0, 20.0, uMax]

xs = copy(vG0)
xs[1] == 0.0 ? (xvec = 2:nc0) : (xvec = 1:nc0)
xs = copy(xs[xvec])
ys = copy(fvL[nvlevel0,L1,isp3][xvec])

# Checking the `guess` parameter
cp1 = coefDMv0(L1-1,p[2:3])
p[1] = copy(cp1)
ys_fit0 = modelDM(xs,p)
resid0 = ys_fit0 - ys

# finding the `optimal` parameter
@time  poly = curve_fit(modelDM,xs,ys,p,lower=lbs, upper=ubs)
# To estimate errors on the fit parameters: `sigm = √(abs(diag(covar)))`
p = poly.param
rn = norm(poly.resid)
sigma = stderror(poly)  # Standard error (stderror) of each parameter by estimating errors on the fit parameters.
modelDMv0(v,p) = p[1] * modelDMexpv0(v,p[2:3],ℓ)

ys_fit = modelDM(xs,p)
resid = ys_fit - ys
