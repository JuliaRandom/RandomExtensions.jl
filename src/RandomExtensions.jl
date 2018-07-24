module RandomExtensions

export Combine, Uniform, Normal, Exponential, CloseOpen, Rand

import Random: Sampler, rand, rand!, gentype

using Random
using Random: GLOBAL_RNG, SamplerTrivial, SamplerSimple, SamplerTag, Repetition

using SparseArrays: sprand, sprandn


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
* a `Combine` object, which can be either `Combine(Pair, S1, S2)` or `Combine(Complex, S1, S2)`,
  where `S1` and `S2` are one of the specifications above; `Pair` or `Complex` can optionally be
  given as concrete types, e.g. `Combine(ComplexF64, 1:3, Int)` to generate `ComplexF64` instead
  of `Complex{Int}`.

`S` usually defaults to [`Float64`](@ref).

If `C...` is not specified, `rand` produces a scalar. Otherwise, `C...` can be:

* a set of integers, or a tuple of `Int`, which specify the dimensions of an `Array` to generate;
* `(p::AbstractFloat, m::Integer, [n::Integer])`, which produces a sparse array of dimensions `(m, n)`,
  in which the probability of any element being nonzero is independently given by `p`
* `(String, [n=8])`, which produces a random `String` of length `n`; the generated string consists of `Char`
  taken from a predefined set like `randstring`, and can be specified with the `S` parameter.
* `(Dict, n)`, which produces a `Dict` of length `n`; `S` must then specify the type of its elements,
  e.g. `Combine(Pair, Int, 2:3)`;
* `(Set, n)`, which produces a `Set` of length `n`;
* `(BitArray, dims...)`, which produces a `BitArray` with the specified dimensions.

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
