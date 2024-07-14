
using Pkg

pknames1 = ["Plots","PyPlot","DataFrames","CSV", "Format","OhMyREPL", "Parsers"]
pknames2 = ["AssociatedLegendrePolynomials","ChebyshevApprox","FastTransforms","SavitzkyGolay"]
pknames3 = ["GaussQuadrature","FastGaussQuadrature","QuadGK","Romberg","Trapz"]
pknames4 = ["SpecialFunctions","HypergeometricFunctions"]
pknames5 = ["LinearAlgebra", "LinearAlgebraX","LeastSquaresOptim","ToeplitzMatrices"]
pknames6 = ["ForwardDiff", "FiniteDifferences"]
pknames7 = ["Loess","Dierckx", "DataInterpolations", "NumericalIntegration","SmoothingSplines"]
pknames8 = ["OrdinaryDiffEq","DifferentialEquations","DifferentialEquations","DiffEqCallbacks","IterativeSolvers","NLsolve"]
pknames9 = ["Latexify","LaTeXStrings","Clustering","KahanSummation"]  # LaTeXStrings
pknames10 = ["StableRNGs","Distances","ModelingToolkit","DataDrivenDiffEq"]
pknames = [pknames1; pknames2; pknames3;pknames4; pknames5;pknames6;pknames7;pknames8;pknames9;pknames10]
# pknames = [pknames4; pknames5;pknames6;pknames7;pknames8]
# pknames = [pknames9]
# pknames = ["StableRNGs"]
for pkname in pknames
    import Pkg; Pkg.add(pkname)
end

# failure

import Pkg; Pkg.add("DiffEqCallbacks")
# import Pkg; Pkg.add("FastKmeans") 
