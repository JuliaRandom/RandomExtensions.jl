# RandomExtensions

[![Build Status](https://travis-ci.org/rfourquet/RandomExtensions.jl.svg?branch=master)](https://travis-ci.org/rfourquet/RandomExtensions.jl)
[![Coverage Status](https://coveralls.io/repos/rfourquet/RandomExtensions.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/rfourquet/RandomExtensions.jl?branch=master)
[![codecov.io](http://codecov.io/github/rfourquet/RandomExtensions.jl/coverage.svg?branch=master)](http://codecov.io/github/rfourquet/RandomExtensions.jl?branch=master)

This package explores a possible extension of `rand`-related
functionalities (from the `Random` module); the code is initially
taken from https://github.com/JuliaLang/julia/pull/24912.
Note that type piracy is commited!
While hopefully useful, this package is still experimental, and
hence unstable. Design or implementation contributions are welcome.

This does mainly 3 things:

1) define distribution objects, to give first-class status to features
   provided by `Random`; for example `rand(Normal(), 3)` is equivalent
   to `randn(3)`; other available distributions: `Exponential`,
   `CloseOpen` (for generation of floats in a close-open range),
   `Uniform` (which can wrap an implicit uniform distribution),
   `make` (to combine distribution for objects made of multiple
   scalars, like `Pair`, `Tuple`, or `Complex`, or for containers);

2) define generation of some containers filled with random values
   (like `Set`, `Dict`, `SparseArray`, `String`, `BitArray`);

3) define a `Rand` iterator, which produces lazily random values.


Point 1) defines a `Distribution` type which is incompatible with the
"Distributions.jl" package. Input on how to unify the two approaches is
welcome.
Point 2) goes somewhat against the trend in `Base` to create
containers using their constructors -- which by the way may be
achieved with the `Rand` iterator from point 3).
Still, I like the terser approach here, as it simply generalizes
to other containers the __current__ `rand` API creating arrays.
See the issue linked above for a discussion on those topics.

For convenience, the following objects from `Random` are re-exported
in this package: `rand!`, `AbstractRNG`, `MersenneTwister`,
`RandomDevice` (`rand` is in `Base`). Functions like `randn!` or
`bitrand` are considered to be obsoleted by this package so are not
re-exported. It's still needed to import `Random` separately in order
to use functions which don't extend the `rand` API, namely
`randsubseq`, `shuffle`, `randperm`, `randcycle`, and their mutating
variants.


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

julia> rand(make(Pair, 1:10, Normal())) # random Pair, where both members have distinct distributions
5 => 0.674375

julia> rand(make(Pair{Number, Any}, 1:10, Normal())) # specify the Pair type
Pair{Number,Any}(1, -0.131617)

julia> rand(Pair{Float64,Int}) # equivalent to rand(make(Pair, Float64, Int))
0.321676 => -4583276276690463733

julia> rand(make(Tuple, 1:10, Normal()))
(9, 1.3407309364427373)

julia> rand(Tuple{Float64,Int}) # equivalent to rand(make(Tuple, Float64, Int))
(0.9830769470405203, -6048436354564488035)

julia> rand(make(NTuple{3}, 1:10)) # produces a 3-tuple with values from 1:10
(5, 9, 6)

julia> rand(make(NTuple{3}, make(Pair, 1:9, Bool))) # make calls can be nested
(2 => false, 8 => true, 7 => false)

julia> rand(make(Complex, Normal())) # each coordinate is drawn from the normal distribution
1.5112317924121632 + 0.723463453534426im

julia> rand(make(Complex, Normal(), 1:10)) # distinct distributions
1.096731587266045 + 8.0im

julia> rand(Normal(ComplexF64)) # equivalent to randn(ComplexF64)
0.9322376894079347 + 0.2812214248483498im

julia> rand(Set, 3)
Set([0.717172, 0.78481, 0.86901])

julia> rand(1:9, Set, 3)
Set([3, 5, 8])

julia> rand(make(Pair, 1:9, Normal()), Dict, 3)
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

julia> rand("123", String, 4) # String considered as a container
"2131"

julia> rand(String, Set, 3) # String considered as a scalar
Set(["0Dfqj6Yr", "ILngfcRz", "HT5IEyK3"])

julia> rand(make(String, 3, "123"))
"211"

julia> rand(BitArray, 3) # equivalent to bitrand(3)
3-element BitArray{1}:
  true
  true
 false

julia> julia> rand(Bernoulli(0.2), BitVector, 10) # using the Bernoulli distribution
10-element BitArray{1}:
 false
 false
 false
 false
  true
 false
  true
 false
 false
  true

julia> rand(1:3, NTuple{3}) # NTuple{3} considered as a container, equivalent to rand(make(NTuple{3}, 1:3))
(3, 3, 1)

julia> Set(Iterators.take(Rand(RandomDevice(), 1:10), 3)) # RNG defaults to Random.GLOBAL_RNG
Set([9, 2, 6])

julia> collect(Iterators.take(Uniform(1:10), 3)) # distributions can be iterated over, using Random.GLOBAL_RNG implicitly
3-element Array{Int64,1}:
  7
 10
  5
```

In some cases, the `Rand` iterator can provide some efficiency gains compared to
repeated calls to `rand`, as it uses the same mechanism as non-scalar generation.
For example, given `a = zeros(10000)`,
`a .+ Rand(1:1000).()` will be faster than `a .+ rand.(Ref(1:1000))`.

Note: as seen in the examples above, `String` can be considered as a scalar or as a container (in the `rand` API).
In a call like `rand(String)`, both APIs coincide, but in `rand(String, 3)`, should we construct a `String` of
length `3` (container API), or an array of strings of default length `8` ? Currently, the package chooses
the first interpretation, partly because it was the first implemented, and also because it may actually be the one
most useful (and offers the tersest API to compete with `randstring`).
But as this package is still unstable, this choice may be revisited in the future.
Note that it's easy to get the result of the second interpretation via either `rand(make(String), 3)`,
`rand(String, (3,))` or `rand(String, Vector, 3)`.
