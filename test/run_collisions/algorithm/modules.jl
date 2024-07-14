
    # using Statistics
    # using ModelingToolkit, DataDrivenDiffEq
    using LinearAlgebra, LinearAlgebraX, ToeplitzMatrices
    using KahanSummation    #  vec = [1.0, 1.0e16, -1.0e16, -0.5]
    using AssociatedLegendrePolynomials, ChebyshevApprox
    using QuadGK, Romberg, Trapz
    using GaussQuadrature      # `datatype` could be anything which is belong to real number;
    # Where `datatype = Float64`, symmetry of vector `μ` will be lost.
    using FastGaussQuadrature  # Accuracy of the results is better than `GaussQuadrature.jl` when `datatype = Float64`
    # and symmetry of vector `μ` is maintained.
    using FastTransforms   # Clenshaw-Curtis and Fejer's quadrature, Padua transform
    export AbstractFFTs, FFTW, FastTransforms, Frequencies, GenericFFT, ann2cxf, associatedjac2jac, bfft, bfft!, brfft, cheb2jac, cheb2leg, cheb2rectdisk, cheb2tet, cheb2tri, cheb2ultra, chebyshevpoints, chebyshevtransform, chebyshevtransform!, chebyshevutransform, chebyshevutransform!, clenshawcurtisnodes, clenshawcurtisweights, cxf2ann, cxf2disk, dct, dct!, disk2cxf, diskones, diskrand, diskrandn, diskzeros, fejernodes1, fejernodes2, fejerweights1, fejerweights2, fft, fft!, fftdims, fftfreq, fftshift, fftshift!, fourier2sph, fourier2sphv, gaunt, ichebyshevtransform, ichebyshevtransform!, ichebyshevutransform, ichebyshevutransform!, idct, idct!, ifft, ifft!, ifftshift, ifftshift!, inufft1, inufft2, ipaduatransform, ipaduatransform!, irfft, iweightedhermitetransform, jac2cheb, jac2jac, jac2ultra, lag2lag, leg2cheb, modifiedherm2herm, modifiedjac2jac, modifiedlag2lag, nufft, nufft1, nufft2, nufft3, paduapoints, paduatransform, paduatransform!, plan_ann2cxf, plan_annulus_analysis, plan_annulus_synthesis, plan_associatedjac2jac, plan_bfft, plan_bfft!, plan_brfft, plan_cheb2jac, plan_cheb2leg, plan_cheb2ultra, plan_chebyshevtransform, plan_chebyshevtransform!, plan_chebyshevutransform, plan_chebyshevutransform!, plan_clenshawcurtis, plan_dct, plan_dct!, plan_disk2cxf, plan_disk_analysis, plan_disk_synthesis, plan_fejer1, plan_fejer2, plan_fft, plan_fft!, plan_ichebyshevtransform, plan_ichebyshevtransform!, plan_ichebyshevutransform, plan_ichebyshevutransform!, plan_idct, plan_idct!, plan_ifft, plan_ifft!, plan_inufft1, plan_inufft2, plan_ipaduatransform!, plan_irfft, plan_jac2cheb, plan_jac2jac, plan_jac2ultra, plan_lag2lag, plan_leg2cheb, plan_modifiedherm2herm, plan_modifiedjac2jac, plan_modifiedlag2lag, plan_nufft, plan_nufft1, plan_nufft2, plan_nufft3, plan_paduatransform!, plan_rectdisk2cheb, plan_rectdisk_analysis, plan_rectdisk_synthesis, plan_rfft, plan_sph2fourier, plan_sph_analysis, plan_sph_synthesis, plan_sphv2fourier, plan_sphv_analysis, plan_sphv_synthesis, plan_spinsph2fourier, plan_spinsph_analysis, plan_spinsph_synthesis, plan_tet2cheb, plan_tet_analysis, plan_tet_synthesis, plan_tri2cheb, plan_tri_analysis, plan_tri_synthesis, plan_ultra2cheb, plan_ultra2jac, plan_ultra2ultra, rectdisk2cheb, rectdiskones, rectdiskrand, rectdiskrandn, rectdiskzeros, rfft, rfftfreq, sph2fourier, sphevaluate, sphones, sphrand, sphrandn, sphv2fourier, sphvones, sphvrand, sphvrandn, sphvzeros, sphzeros, spinsphones, spinsphrand, spinsphrandn, spinsphzeros, tet2cheb, tetones, tetrand, tetrandn, tetzeros, tri2cheb, trievaluate, triones, trirand, trirandn, trizeros, ultra2cheb, ultra2jac, ultra2ultra, weightedhermitetransform
    using SpecialFunctions, HypergeometricFunctions #, SavitzkyGolay
    # using FastChebInterp, BasicInterpolators
    # using KahanSummation
    # ################################################## Interpolations and Smoothing
    using Dierckx
    # using SpecialPolynomials, Polynomials
    using LeastSquaresOptim                    # Better than `LsqFit.jl`
    using NLsolve                              # 
    # using LsqFit
    # using BasisFunctionExpansions
    # using Loess, Lowess, GaussNewton
    # using NumericalIntegration
    # using DataInterpolations, BasicInterpolators
    # using SmoothingSplines 
    # using Preconditioners
    # using SparseArrays
    # using IncompleteLU, IncompleteSe
    using ForwardDiff, FiniteDifferences
    using DifferentialEquations, OrdinaryDiffEq, DiffEqCallbacks
    # using DiffEqDevTools
    # using IRKGaussLegendre
    # using GeometricIntegrators
    # using IterativeSolvers
    # using Xsum, AccurateArithmetic          # for higher precision
    import FastTransforms: chebyshevmoments1
    # import DoubleFloats.Double64
    # using GenericLinearAlgebra   # `svd!()` instead of the one in Base.jl for `pinv()` with less accuracy when BigFloat is used.
    # import GenericLinearAlgebra.pinv
    # using MatrixEquations
    # using DoubleFloats
    # import Base.LinearAlgebra.pinv
    # Richardson
    # using Query , DataFramesMeta, NumericIO

    # using MatrixEquations, IterativeSolvers
    # using SylvesterEquations
    using OhMyREPL
    using Plots
    using DataFrames, Format, CSV, Parsers
    # using DelimitedFiles    
    using LaTeXStrings 
    # using Latexify
    