# RandomExtensions

[![Build Status](https://travis-ci.org/rfourquet/RandomExtensions.jl.svg?branch=master)](https://travis-ci.org/rfourquet/RandomExtensions.jl)
[![Coverage Status](https://coveralls.io/repos/rfourquet/RandomExtensions.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/rfourquet/RandomExtensions.jl?branch=master)
[![codecov.io](http://codecov.io/github/rfourquet/RandomExtensions.jl/coverage.svg?branch=master)](http://codecov.io/github/rfourquet/RandomExtensions.jl?branch=master)

This package explores a possible extension of `rand`-related
functionalities (from the `Random` module); the code is initially
taken from https://github.com/JuliaLang/julia/pull/24912.
Note that type piracy is commited!
While hopefully useful, this package is still very experimental, and
hence unstable.

This does mainly 3 things:

1) define distribution objects, to give first-class status to features
   provided by `Random`; for example `rand(Normal(), 3)` is equivalent
   to `randn(3)`; other available distributions: `Exponential`,
   `CloseOpen` (for generation of floats in a close-open range),
   `Uniform` (which can wrap an implicit uniform distribution),
   `Combine` (to "combine" distribution for objects made of multiple
   scalars, like `Pair` or `Complex`);


2) define generation of some containers filled with random values
   (like `Set`, `Dict`, `SparseArray`, `String`, `BitArray`);

3) define a `Rand` iterator, which produces lazily random values.


There is not much documentation for now: `rand`'s docstring is updated,
and here are some examples:

```julia
julia> rand(CloseOpen()) # like rand(Float64)
0.7678877639669386

julia> rand(CloseOpen(1.0, 10.0)) # generation in [1.0, 10.0)
4.309057677479184

julia> rand(Normal(0.0, 10.0)) # explicit μ and σ parameters
-8.473790458128912

julia> rand(Uniform(1:3)) # equivalent to rand(1:3)
2

julia> rand(Combine(Pair, 1:10, Normal())) # random Pair, where both members have distinct distributions
5 => 0.674375

julia> rand(Combine(Pair{Number, Any}, 1:10, Normal())) # specify the Pair type
Pair{Number,Any}(1, -0.131617)

julia> rand(Combine(Complex, Normal())) # each coordinate is drawn from the normal distribution
1.5112317924121632 + 0.723463453534426im

julia> rand(Combine(Complex, Normal(), 1:10)) # distinct distributions
1.096731587266045 + 8.0im

julia> rand(Set, 3)
Set([0.717172, 0.78481, 0.86901])

julia> rand(1:9, Set, 3)
Set([3, 5, 8])

julia> rand(Combine(Pair, 1:9, Normal()), Dict, 3)
Dict{Int64,Float64} with 3 entries:
  9 => 0.916406
  3 => -2.44958
  8 => -0.703348

julia> rand(0.3, 9) # equivalent to sprand(9, 0.3)
9-element SparseVector{Float64,Int64} with 3 stored entries:
  [1]  =  0.173858
  [6]  =  0.568631
  [8]  =  0.297207

julia> rand(Normal(), 0.3, 2, 3) # equivalent to sprandn(2, 3, 0.3)
2×3 SparseMatrixCSC{Float64,Int64} with 2 stored entries:
  [2, 1]  =  0.448981
  [1, 2]  =  0.730103

julia> rand(String, 4) # equivalent to randstring(4)
"5o75"

julia> rand(BitArray, 3) # equivalent to bitrand(3)
3-element BitArray{1}:
  true
  true
 false

julia> Set(Iterators.take(Rand(RandomDevice(), 1:10), 3)) # RNG defaults to Random.GLOBAL_RNG
Set([9, 2, 6])

julia> collect(Iterators.take(Uniform(1:10), 3)) # distributions can be iterated over, using Random.GLOBAL_RNG implicitly
3-element Array{Int64,1}:
  7
 10
  5
```
