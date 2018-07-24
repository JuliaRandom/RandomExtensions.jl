# generation of some containers filled with random values


## dicts

rand!(A::AbstractDict{K,V}, dist::Distribution{<:Pair}=Combine(Pair, K, V)) where {K,V} =
    rand!(GLOBAL_RNG, A, dist)

rand!(rng::AbstractRNG, A::AbstractDict{K,V},
      dist::Distribution{<:Pair}=Combine(Pair, K, V)) where {K,V} =
          rand!(GLOBAL_RNG, A, Sampler(rng, dist))

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

rand(u::Distribution{<:Pair}, ::Type{T}, n::Integer) where {T<:AbstractDict} = rand(GLOBAL_RNG, u, T, n)


## sets

rand!(A::AbstractSet{T}, X) where {T} = rand!(GLOBAL_RNG, A, X)
rand!(A::AbstractSet{T}, ::Type{X}=T) where {T,X} = rand!(GLOBAL_RNG, A, X)

rand!(rng::AbstractRNG, A::AbstractSet, X) = rand!(rng, A, Sampler(rng, X))
rand!(rng::AbstractRNG, A::AbstractSet{T}, ::Type{X}=T) where {T,X} = rand!(rng, A, Sampler(rng, X))

_rand0!(rng::AbstractRNG, A::AbstractSet, n::Integer, X) = _rand!(rng, A, n, Sampler(rng, X))
_rand0!(rng::AbstractRNG, A::AbstractSet, n::Integer, ::Type{X}) where {X} = _rand!(rng, A, n, Sampler(rng, X))
_rand0!(rng::AbstractRNG, A::AbstractSet, n::Integer, sp::Sampler) = _rand!(rng, A, n, sp)

rand!(rng::AbstractRNG, A::AbstractSet, sp::Sampler) = _rand!(rng, A, length(A), sp)


rand(r::AbstractRNG, ::Type{T}, n::Integer) where {T<:AbstractSet} = rand(r, Float64, T, n)
rand(                ::Type{T}, n::Integer) where {T<:AbstractSet} = rand(GLOBAL_RNG, T, n)

rand(r::AbstractRNG, X, ::Type{T}, n::Integer) where {T<:AbstractSet} = _rand0!(r, deduce_type(T, eltype(X))(), n, X)
rand(                X, ::Type{T}, n::Integer) where {T<:AbstractSet} = rand(GLOBAL_RNG, X, T, n)

rand(r::AbstractRNG, ::Type{X}, ::Type{T}, n::Integer) where {X,T<:AbstractSet} = _rand0!(r, deduce_type(T, X)(), n, X)
rand(                ::Type{X}, ::Type{T}, n::Integer) where {X,T<:AbstractSet} = rand(GLOBAL_RNG, X, T, n)


## sparse vectors & matrices

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
    sprand(r, m, n, p, (r, n)->rand(r, X, n), eltype(X))

rand(r::AbstractRNG, X, p::AbstractFloat, m::Integer, n::Integer) =
    rand(r, Sampler(r, X), p, m, n)

rand(r::AbstractRNG, ::Type{X}, p::AbstractFloat, m::Integer, n::Integer) where {X} =
    rand(r, Sampler(r, X), p, m, n)

rand(X, p::AbstractFloat, m::Integer, n::Integer) = rand(GLOBAL_RNG, X, p, m, n)


## String

let b = UInt8['0':'9';'A':'Z';'a':'z']
    global rand
    rand(rng::AbstractRNG, chars, ::Type{String}, n::Integer=8) = String(rand(rng, chars, n))
    rand(                  chars, ::Type{String}, n::Integer=8) = rand(GLOBAL_RNG, chars, String, n)
    rand(rng::AbstractRNG, ::Type{String}, n::Integer=8) = rand(rng, b, String, n)
    rand(                  ::Type{String}, n::Integer=8) = rand(GLOBAL_RNG, b, String, n)
end


## BitArray

const BitArrays = Union{BitArray,BitVector,BitMatrix}

rand(r::AbstractRNG, ::Type{T}, dims::Dims) where {T<:BitArrays} =
    rand!(r, T(undef, dims))

rand(r::AbstractRNG, ::Type{T}, dims::Integer...) where {T<:BitArrays} =
    rand!(r, T(undef, convert(Dims, dims)))

rand(::Type{T}, dims::Dims) where {T<:BitArrays} =
    rand!(T(undef, dims))

rand(::Type{T}, dims::Integer...) where {T<:BitArrays} =
    rand!(T(undef, convert(Dims, dims)))
