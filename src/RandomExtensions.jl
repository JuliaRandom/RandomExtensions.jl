module RandomExtensions

export make, Uniform, Normal, Exponential, CloseOpen, OpenClose, OpenOpen, CloseClose, Rand, Bernoulli

# re-exports from Random, which don't overlap with new functionality and not from misc.jl
export rand!, AbstractRNG, MersenneTwister, RandomDevice

import Random: Sampler, rand, rand!, gentype

using Random
using Random: GLOBAL_RNG, SamplerTrivial, SamplerSimple, SamplerTag, SamplerType, Repetition

using SparseArrays: sprand, sprandn, AbstractSparseArray, SparseVector, SparseMatrixCSC


## a dummy container type to take advangage of SamplerTag constructor

struct Cont{T} end

Base.eltype(::Type{Cont{T}}) where {T} = T


## some helper functions, not to be overloaded, except default_sampling

default_sampling(::Type{X}) where {X} = error("default_sampling($X) not defined")
default_sampling(::X)       where {X} = default_sampling(X)

default_gentype(::Type{T}) where {T} = val_gentype(default_sampling(T))
default_gentype(::X)       where {X} = default_gentype(X)

val_gentype(X)                   = gentype(X)
val_gentype(::Type{X}) where {X} = X


## includes

include("distributions.jl")
include("sampling.jl")
include("containers.jl")
include("iteration.jl")


## updated rand docstring (TODO: replace Base's one)

"""
    rand([rng=GLOBAL_RNG], [S], [C...]) # RandomExtensions

Pick a random element or collection of random elements from the set of values specified by `S`;
`S` can be

* an indexable collection (for example `1:n` or `['x','y','z']`),
* an `AbstractDict` or `AbstractSet` object,
* a string (considered as a collection of characters), or
* a type: the set of values to pick from is then equivalent to `typemin(S):typemax(S)` for
  integers (this is not applicable to [`BigInt`](@ref)), and to ``[0, 1)`` for floating
  point numbers;
* a `Distribution` object, e.g. `Normal()` for a normal distribution (like `randn()`),
  or `CloseOpen(10.0, 20.0)` for uniform `Float64` numbers in the range ``[10.0, 20.0)``;
* a `make` object, which can be e.g. `make(Pair, S1, S2)` or `make(Complex, S1, S2)`,
  where `S1` and `S2` are one of the specifications above; `Pair` or `Complex` can optionally be
  given as concrete types, e.g. `make(ComplexF64, 1:3, Int)` to generate `ComplexF64` instead
  of `Complex{Int}`.

`S` usually defaults to [`Float64`](@ref).

If `C...` is not specified, `rand` produces a scalar. Otherwise, `C` can be:

* a set of integers, or a tuple of `Int`, which specify the dimensions of an `Array` to generate;
* `(Array, dims...)`: same as above, but with `Array` specified explicitely;
* `(p::AbstractFloat, m::Integer, [n::Integer])`, which produces a sparse array of dimensions `(m, n)`,
  in which the probability of any element being nonzero is independently given by `p`;
* `(String, [n=8])`, which produces a random `String` of length `n`; the generated string consists of `Char`
  taken from a predefined set like `randstring`, and can be specified with the `S` parameter;
* `(Dict, n)`, which produces a `Dict` of length `n`; if `Dict` is given without type parameters,
  then `S` must be specified;
* `(Set, n)` or `(BitSet, n)`, which produces a set of length `n`;
* `(BitArray, dims...)`, which produces a `BitArray` with the specified dimensions.

For `Array`, `Dict` and `Set`, a less abstract type can be specified, e.g. `Set{Float64}`, to force
the type of the result regardless of the `S` parameter. In particular, in the absence of `S`, the
type parameter(s) of the container play the role of `S`; for example, `rand(Dict{Int,Float64}, n)`
is equivalent to `rand(make(Pair, Int, Float64), Dict, n)`.

# Examples
```julia-repl
julia> rand(Int, 2)
2-element Array{Int64,1}:
 1339893410598768192
 1575814717733606317

julia> rand(MersenneTwister(0), Dict(1=>2, 3=>4))
1=>2

julia> rand("abc", String, 12)
"bccaacaabaac"

julia> rand(1:10, Set, 3)
Set([3, 8, 6])

julia> rand(1:10, Set{Float64}, 3)
Set([10.0, 5.0, 6.0])
```

!!! note
    The complexity of `rand(rng, s::Union{AbstractDict,AbstractSet})`
    is linear in the length of `s`, unless an optimized method with
    constant complexity is available, which is the case for `Dict`,
    `Set` and `BitSet`. For more than a few calls, use `rand(rng,
    collect(s))` instead, or either `rand(rng, Dict(s))` or `rand(rng,
    Set(s))` as appropriate.
"""
rand

end # module
