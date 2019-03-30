# RandomExtensions

[![Build Status](https://travis-ci.org/rfourquet/RandomExtensions.jl.svg?branch=master)](https://travis-ci.org/rfourquet/RandomExtensions.jl)
[![Coverage Status](https://coveralls.io/repos/rfourquet/RandomExtensions.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/rfourquet/RandomExtensions.jl?branch=master)
[![codecov.io](http://codecov.io/github/rfourquet/RandomExtensions.jl/coverage.svg?branch=master)](http://codecov.io/github/rfourquet/RandomExtensions.jl?branch=master)

This package explores a possible extension of `rand`-related
functionalities (from the `Random` module); the code is initially
taken from https://github.com/JuliaLang/julia/pull/24912.
Note that type piracy is committed!
While hopefully useful, this package is still experimental, and
hence unstable. User feedback, and design or implementation contributions are welcome.

This does essentially 4 things:

1) define distribution objects, to give first-class status to features
   provided by `Random`; for example `rand(Normal(), 3)` is equivalent
   to `randn(3)`; other available distributions: `Exponential`,
   `CloseOpen` (for generation of floats in a close-open range) and friends,
   `Uniform` (which can wrap an implicit uniform distribution);

2) define `make` methods, which can combine distributions for objects made of multiple scalars, like
   `Pair`, `Tuple`, or `Complex`, or describe how to generate more complex objects, like containers;

3) extend the `rand([rng], [S], dims)` API to allow the generation of other containers than arrays
   (like `Set`, `Dict`, `SparseArray`, `String`, `BitArray`);

4) define a `Rand` iterator, which produces lazily random values.


Point 1) defines a `Distribution` type which is incompatible with the
"Distributions.jl" package. Input on how to unify the two approaches is
welcome.

Point 2) is really the core of this package. `make` provides a vocabulary to define the generation
of "scalars" which require more than one argument to be described, e.g. pairs from `1:3` to `Int`
(`rand(make(Pair, 1:3, Int))`) or regular containers (e.g. `make(Array, 2, 3)`). The point of
calling `make` rather than putting all the arguments in `rand` directly is simplicity and
composability: the `make` call always occurs as the second argument to `rand` (or first if the RNG
is omitted). For example, `rand(make(Array, 2, 3), 3)` creates an array of matrices.
Of course, `make` is not necessary, in that the same can be achieved with an ad hoc `struct`,
which in some cases is clearer (e.g. `Normal(m, s)` rather than something like `make(Float64, Val(:Normal), m, s)`).

Point 3) allows something like `rand(1:30, Set, 10)` to produce a `Set` of length `10` with values
from `1:30`. The idea is that `rand([rng], [S], Cont, etc...)` should always be equivalent to
`rand([rng], make(Cont, [S], etc...))`. This design goes somewhat against the trend in `Base` to create
containers using their constructors -- which by the way may be achieved via the `Rand` iterator from
point 4). Still, I like the terse approach here, as it simply generalizes to other containers the
_current_ `rand` API creating arrays. See the issue linked above for a discussion on these topics.

For convenience, the following names from `Random` are re-exported
in this package: `rand!`, `AbstractRNG`, `MersenneTwister`,
`RandomDevice` (`rand` is in `Base`). Functions like `randn!` or
`randstring` are considered to be obsoleted by this package so are not
re-exported. It's still needed to import `Random` separately in order
to use functions which don't extend the `rand` API, namely
`randsubseq`, `shuffle`, `randperm`, `randcycle`, and their mutating
variants.


There is not much documentation for now: `rand`'s docstring is updated,
and here are some examples:

```julia
julia> rand(CloseOpen(Float64)) # equivalent to rand(Float64)
0.7678877639669386

julia> rand(CloseClose(1.0f0, 10)) # generation in [1.0f0, 10.0f0]
6.62467f0

julia> rand(OpenOpen(2.0^52, 2.0^52+1)) == 2.0^52 # exactness not guaranteed for "unreasonable" values!
true

julia> rand(Normal(0.0, 10.0)) # explicit μ and σ parameters
-8.473790458128912

julia> rand(Uniform(1:3)) # equivalent to rand(1:3)
2

julia> rand(make(Pair, 1:10, Normal())) # random Pair, where both members have distinct distributions
5 => 0.674375

julia> rand(make(Pair{Number,Any}, 1:10, Normal())) # specify the Pair type
Pair{Number,Any}(1, -0.131617)

julia> rand(Pair{Float64,Int}) # equivalent to rand(make(Pair, Float64, Int))
0.321676 => -4583276276690463733

julia> rand(make(Tuple, 1:10, UInt8, OpenClose()))
(9, 0x6b, 0.34900083923775505)

julia> rand(Tuple{Float64,Int}) # equivalent to rand(make(Tuple, Float64, Int))
(0.9830769470405203, -6048436354564488035)

julia> rand(make(NTuple{3}, 1:10)) # produces a 3-tuple with values from 1:10
(5, 9, 6)

julia> rand(make(NTuple{N,UInt8} where N, 1:3, 5))
(0x02, 0x03, 0x02, 0x03, 0x02)

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

julia>  rand!(ans, Exponential())
Set([0.7935073925105659, 2.593684878770254, 1.629181233597078])

julia> rand(1:9, Set, 3) # if you try `rand(1:3, Set, 9)`, it will take a while ;-)
Set([3, 5, 8])

julia> rand(Dict{String,Int8}, 2)
Dict{String,Int8} with 3 entries:
  "vxybIbae" => 42
  "bO2fTwuq" => -13

julia> rand(make(Pair, 1:9, Normal()), Dict, 3)
Dict{Int64,Float64} with 3 entries:
  9 => 0.916406
  3 => -2.44958
  8 => -0.703348

julia> rand(SparseVector, 0.3, 9) # equivalent to sprand(9, 0.3)
9-element SparseVector{Float64,Int64} with 3 stored entries:
  [1]  =  0.173858
  [6]  =  0.568631
  [8]  =  0.297207

julia> rand(Normal(), SparseMatrixCSC, 0.3, 2, 3) # equivalent to sprandn(2, 3, 0.3)
2×3 SparseMatrixCSC{Float64,Int64} with 2 stored entries:
  [2, 1]  =  0.448981
  [1, 2]  =  0.730103

# like for Array, sparse arrays enjoy to be special cased: `SparseVector` or `SparseMatrixCSC` can be omitted:

julia> rand(make(make(1:9, 0.3, 2, 3), 0.1, 4)) # possible, bug ugly output when non-empty :-/
4-element SparseVector{SparseMatrixCSC{Int64,Int64},Int64} with 0 stored entries

julia> rand(String, 4) # equivalent to randstring(4)
"5o75"

julia> rand("123", String, 4) # like above, String creation with the "container" syntax ...
"2131"

julia> rand(make(String, 3, "123")) # ... which is as always equivalent to a call to make
"211"

julia> rand(String, Set, 3) # String considered as a scalar
Set(["0Dfqj6Yr", "ILngfcRz", "HT5IEyK3"])

julia> rand(BitArray, 3) # equivalent to, but unfortunately more verbose than, bitrand(3)
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

julia> rand(1:3, Tuple{Int,UInt8, BigFloat}) # works also with more general tuple types ...
(3, 0x02, 2.0)

julia> rand(1:3, NamedTuple{(:a, :b)}) # ... and with named tuples
(a = 3, b = 2)

julia> RandomExtensions.random_staticarrays() # poor man's conditional modules!
# ugly warning

julia> rand(make(MVector{2,AbstractString}, String), SMatrix{3, 2})
3×2 SArray{Tuple{3,2},MArray{Tuple{2},AbstractString,1,2},2,6} with indices SOneTo(3)×SOneTo(2):
 ["SzPKXHFk", "1eFXaUiM"]  ["RJnHwhb7", "jqfLcY8a"]
 ["FMTKcBY8", "eoYtNntD"]  ["FzdD530L", "ux6sWGMU"]
 ["fFJuUtJQ", "H2mAQrIV"]  ["pt0OYFJw", "O0fCfjjR"]

julia> Set(Iterators.take(Rand(RandomDevice(), 1:10), 3)) # RNG defaults to Random.GLOBAL_RNG
Set([9, 2, 6]) # note that the set could end up with less than 3 elements if `Rand` generates duplicates

julia> collect(Iterators.take(Uniform(1:10), 3)) # distributions can be iterated over, using Random.GLOBAL_RNG implicitly
3-element Array{Int64,1}:
  7
 10
  5
```

In some cases, the `Rand` iterator can provide efficiency gains compared to
repeated calls to `rand`, as it uses the same mechanism as array generation.
For example, given `a = zeros(1000)` and `s = BitSet(1:1000)`,
`a .+ Rand(s).()` is three times faster than `a .+ rand.(Ref(s))`.

Note: as seen in the examples above, `String` can be considered as a scalar or as a container (in the `rand` API).
In a call like `rand(String)`, both APIs coincide, but in `rand(String, 3)`, should we construct a `String` of
length `3` (container API), or an array of strings of default length `8` ? Currently, the package chooses
the first interpretation, partly because it was the first implemented, and also because it may actually be the one
most useful (and offers the tersest API to compete with `randstring`).
But as this package is still unstable, this choice may be revisited in the future.
Note that it's easy to get the result of the second interpretation via either `rand(make(String), 3)`,
`rand(String, (3,))` or `rand(String, Vector, 3)`.

How to extend: the `make` function is meant to be extensible, and there are some helper functions
which make it easy, but the internals are not fully settled. By default, `make(T, args...)` will
create a `Make{find_type(T, args...)}` object, say `m`, which contain `args...` as fields. For type
stable code, the `rand` machinery likes to know the exact type of the object which will be generated by
`rand(m)`, and `find_type(T, args...)` is supposed to return that type. For example,
`find_type(Pair, 1:3, UInt) == Pair{Int,UInt}`.
Then just define `rand` for `m` like documented in the `Random` module, e.g.
`rand(rng::AbstractRNG, sp::SamplerTrivial{<:Make{P}}) where {P<:Pair} = P(rand(sp[].x), rand(sp[].y))`.

This package started out of frustration with the limitations of the `Random` module. Besides
generating simple scalars and arrays, very little is supported out of the box. For example,
generating a random `Dict` is too complex. Moreover, there are too many functions for my taste:
`rand`, `randn`, `randexp`, `sprand` (with its exotic `rfn` parameter), `sprandn`, ~~`sprandexp`~~,
`randstring`, `bitrand`, and mutating counterparts (but I believe `randn` will never go away, as
it's so terse). I hope that this package can serve as a starting point towards improving `Random`.
