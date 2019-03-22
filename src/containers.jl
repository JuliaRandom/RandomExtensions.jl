# generation of some containers filled with random values


## arrays (same as in Random, but with explicit type specification, e.g. rand(Int, Array, 4)

check_dims(A::Type{<:AbstractArray{T,N} where T}, dims::Dims) where {N} =
    length(dims) == N ? dims : throw(DomainError(dims, "incompatible dimensions for $A"))
check_dims(::Type{<:AbstractArray}, dims::Dims) = dims

# cf. inference bug https://github.com/JuliaLang/julia/issues/28762
# we have to write out all combinations for getting proper inference
array_type(::Type{Array{T}}, ::Type{X}) where {T,X} = T
array_type(::Type{Array{T,N} where T}, ::Type{X}) where {N,X} = X
array_type(::Type{Array{T,N}}, ::Type{X}) where {T,N,X} = T
array_type(::Type{Array}, ::Type{X}) where {X} = X

make_array(A::Type{<:Array}, ::Type{X}, dims::Dims) where {X} = Array{array_type(A, X)}(undef, check_dims(A, dims))

default_sampling(::Type{A}) where {A<:Array} = array_type(A, Float64)

rand(r::AbstractRNG, A::Type{<:Array}, dims::Dims) = rand(r, default_sampling(A), A, dims)
rand(                A::Type{<:Array}, dims::Dims) = rand(GLOBAL_RNG, default_sampling(A), A, dims)

rand(r::AbstractRNG, A::Type{<:Array}, dims::Integer...) = rand(r, default_sampling(A), A, Dims(dims))
rand(                A::Type{<:Array}, dims::Integer...) = rand(GLOBAL_RNG, default_sampling(A), A, Dims(dims))

rand(r::AbstractRNG, X, A::Type{<:Array}, dims::Dims) = rand!(r, make_array(A, gentype(X), dims), X)
rand(                X, A::Type{<:Array}, dims::Dims) = rand(GLOBAL_RNG, X, A, dims)

rand(r::AbstractRNG, X, A::Type{<:Array}, dims::Integer...) = rand(r, X, A, Dims(dims))
rand(                X, A::Type{<:Array}, dims::Integer...) = rand(GLOBAL_RNG, X, A, Dims(dims))

rand(r::AbstractRNG, ::Type{X}, A::Type{<:Array}, dims::Dims) where {X} = rand!(r, make_array(A, X, dims), X)
rand(                ::Type{X}, A::Type{<:Array}, dims::Dims) where {X} = rand(GLOBAL_RNG, X, A, dims)

rand(r::AbstractRNG, ::Type{X}, A::Type{<:Array}, dims::Integer...) where {X} = rand(r, X, A, Dims(dims))
rand(                ::Type{X}, A::Type{<:Array}, dims::Integer...) where {X} = rand(GLOBAL_RNG, X, A, Dims(dims))


## dicts

# again same inference bug
# TODO: extend to AbstractDict ? (needs to work-around the inderence bug)
default_sampling(::Type{<:Dict{K,V}}) where {K,V} = Pair{K,V}
default_sampling(D::Type{<:Dict}) = throw(ArgumentError("under-specified scalar type for $D"))

rand!(A::AbstractDict{K,V}, dist::Union{Type{<:Pair},Distribution{<:Pair}}=Combine(Pair, K, V)) where {K,V} =
    rand!(GLOBAL_RNG, A, dist)

rand!(rng::AbstractRNG, A::AbstractDict{K,V},
      dist::Union{Type{<:Pair},Distribution{<:Pair}}=Combine(Pair, K, V)) where {K,V} =
          rand!(rng, A, Sampler(rng, dist))

function _rand!(rng::AbstractRNG, A::Union{AbstractDict,AbstractSet}, n::Integer, sp::Sampler)
    empty!(A)
    while length(A) < n
        push!(A, rand(rng, sp))
    end
    A
end

rand!(rng::AbstractRNG, A::AbstractDict{K,V}, sp::Sampler) where {K,V} = _rand!(rng, A, length(A), sp)

rand(rng::AbstractRNG, dist::Distribution{P}, ::Type{T}, n::Integer) where {P<:Pair,T<:AbstractDict} =
    _rand!(rng, deduce_type(T, fieldtype(P, 1), fieldtype(P, 2))(), n, Sampler(rng, dist))

rand(rng::AbstractRNG, ::Type{P}, ::Type{T}, n::Integer) where {P<:Pair,T<:AbstractDict} = rand(rng, Uniform(P), T, n)

rand(rng::AbstractRNG, ::Type{T}, n::Integer) where {T<:AbstractDict} = rand(rng, default_sampling(T), T, n)

rand(u::Distribution{<:Pair}, ::Type{T}, n::Integer) where {T<:AbstractDict} = rand(GLOBAL_RNG, u, T, n)

rand(::Type{P}, ::Type{T}, n::Integer) where {P<:Pair,T<:AbstractDict} = rand(GLOBAL_RNG, Uniform(P), T, n)

rand(::Type{T}, n::Integer) where {T<:AbstractDict} = rand(GLOBAL_RNG, default_sampling(T), T, n)

## sets

default_sampling(::Type{<:AbstractSet}) = Float64
default_sampling(::Type{<:AbstractSet{T}}) where {T} = T

rand!(A::AbstractSet{T}, X) where {T} = rand!(GLOBAL_RNG, A, X)
rand!(A::AbstractSet{T}, ::Type{X}=T) where {T,X} = rand!(GLOBAL_RNG, A, X)

rand!(rng::AbstractRNG, A::AbstractSet, X) = rand!(rng, A, Sampler(rng, X))
rand!(rng::AbstractRNG, A::AbstractSet{T}, ::Type{X}=T) where {T,X} = rand!(rng, A, Sampler(rng, X))

_rand0!(rng::AbstractRNG, A::AbstractSet, n::Integer, X) = _rand!(rng, A, n, Sampler(rng, X))
_rand0!(rng::AbstractRNG, A::AbstractSet, n::Integer, ::Type{X}) where {X} = _rand!(rng, A, n, Sampler(rng, X))
_rand0!(rng::AbstractRNG, A::AbstractSet, n::Integer, sp::Sampler) = _rand!(rng, A, n, sp)

rand!(rng::AbstractRNG, A::AbstractSet, sp::Sampler) = _rand!(rng, A, length(A), sp)


rand(r::AbstractRNG, ::Type{T}, n::Integer) where {T<:AbstractSet} = rand(r, default_sampling(T), T, n)
rand(                ::Type{T}, n::Integer) where {T<:AbstractSet} = rand(GLOBAL_RNG, T, n)

rand(r::AbstractRNG, X, ::Type{T}, n::Integer) where {T<:AbstractSet} = _rand0!(r, deduce_type(T, gentype(X))(), n, X)
rand(                X, ::Type{T}, n::Integer) where {T<:AbstractSet} = rand(GLOBAL_RNG, X, T, n)

rand(r::AbstractRNG, ::Type{X}, ::Type{T}, n::Integer) where {X,T<:AbstractSet} = _rand0!(r, deduce_type(T, X)(), n, X)
rand(                ::Type{X}, ::Type{T}, n::Integer) where {X,T<:AbstractSet} = rand(GLOBAL_RNG, X, T, n)

### BitSet

default_sampling(::Type{BitSet}) = Int8 # almost arbitrary, may change

Combine(::Type{BitSet}, X, n::Integer) = Combine2{BitSet}(X, Int(n))

Sampler(RNG::Type{<:AbstractRNG}, c::Combine{BitSet}, n::Repetition) =
    SamplerTag{BitSet}((Sampler(RNG, c.x, n), c.y))

function rand(rng::MersenneTwister, sp::SamplerTag{BitSet})
    s = sizehint!(BitSet(), sp.data[2])
    _rand!(rng, s, sp.data[2], sp.data[1])
end


## sparse vectors & matrices

# TODO: implement default_sampling

rand(r::AbstractRNG, p::AbstractFloat, m::Integer) = sprand(r, m, p)
rand(                p::AbstractFloat, m::Integer) = sprand(GLOBAL_RNG, m, p)
rand(r::AbstractRNG, p::AbstractFloat, m::Integer, n::Integer) = sprand(r, m, n, p)
rand(                p::AbstractFloat, m::Integer, n::Integer) = sprand(GLOBAL_RNG, m, n, p)

rand(r::AbstractRNG, X::Sampler, p::AbstractFloat, m::Integer) =
    sprand(r, m, p, (r, n)->rand(r, X, n))

rand(r::AbstractRNG, X, p::AbstractFloat, m::Integer) =
    rand(r, Sampler(r, X), p, m)

rand(r::AbstractRNG, ::Type{X}, p::AbstractFloat, m::Integer) where {X} =
    rand(r, Sampler(r, X), p, m)

rand(X, p::AbstractFloat, m::Integer) = rand(GLOBAL_RNG, X, p, m)

rand(r::AbstractRNG, X::Sampler, p::AbstractFloat, m::Integer, n::Integer) =
    sprand(r, m, n, p, (r, n)->rand(r, X, n), gentype(X))

rand(r::AbstractRNG, X, p::AbstractFloat, m::Integer, n::Integer) =
    rand(r, Sampler(r, X), p, m, n)

rand(r::AbstractRNG, ::Type{X}, p::AbstractFloat, m::Integer, n::Integer) where {X} =
    rand(r, Sampler(r, X), p, m, n)

rand(X, p::AbstractFloat, m::Integer, n::Integer) = rand(GLOBAL_RNG, X, p, m, n)


## String

rand(rng::AbstractRNG, chars, ::Type{String}, n::Integer=8) = rand(rng, Combine(String, chars, n))
rand(                  chars, ::Type{String}, n::Integer=8) = rand(GLOBAL_RNG, Combine(String, chars, n))

rand(rng::AbstractRNG, ::Type{String}, n::Integer=8) = rand(rng, Combine(String, n))
rand(                  ::Type{String}, n::Integer=8) = rand(GLOBAL_RNG, Combine(String, n))


## BitArray

default_sampling(::Type{<:BitArray}) = Bool

const BitArrays = Union{BitArray,BitVector,BitMatrix}

rand(r::AbstractRNG, ::Type{T}, dims::Dims) where {T<:BitArrays} =
    rand!(r, T(undef, dims))

rand(r::AbstractRNG, ::Type{T}, dims::Integer...) where {T<:BitArrays} =
    rand!(r, T(undef, convert(Dims, dims)))

rand(::Type{T}, dims::Dims) where {T<:BitArrays} =
    rand!(T(undef, dims))

rand(::Type{T}, dims::Integer...) where {T<:BitArrays} =
    rand!(T(undef, convert(Dims, dims)))

### with sample information

rand(r::AbstractRNG, X, ::Type{T}, dims::Dims) where {T<:BitArrays} =
    rand!(r, T(undef, dims), X)

rand(r::AbstractRNG, X, ::Type{T}, dims::Integer...) where {T<:BitArrays} =
    rand!(r, T(undef, Dims(dims)), X)

rand(X, ::Type{T}, dims::Dims) where {T<:BitArrays} =
    rand!(T(undef, dims), X)

rand(X, ::Type{T}, dims::Integer...) where {T<:BitArrays} =
    rand!(T(undef, convert(Dims, dims)), X)


## NTuple as a container

rand(r::AbstractRNG, X,         ::Type{NTuple{N}}) where {N}   = rand(r,          Combine(NTuple{N}, X))
rand(                X,         ::Type{NTuple{N}}) where {N}   = rand(GLOBAL_RNG, Combine(NTuple{N}, X))
rand(r::AbstractRNG, ::Type{X}, ::Type{NTuple{N}}) where {X,N} = rand(r,          Combine(NTuple{N}, X))
rand(                ::Type{X}, ::Type{NTuple{N}}) where {X,N} = rand(GLOBAL_RNG, Combine(NTuple{N}, X))

### disambiguate

rand(::AbstractRNG, ::Type{Tuple{}}) = ()
