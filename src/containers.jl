# generation of some containers filled with random values


default_sampling(::Type{X}) where {X} = error("default_sampling($X) not defined")
default_sampling(x::X) where {X} = default_sampling(X)


## arrays (same as in Random, but with explicit type specification, e.g. rand(Int, Array, 4)

macro make_array_container(Cont)
    definitions =
        [ :(rand(r::AbstractRNG,            T::Type{<:$Cont}, dims::Dims) =                 rand(r,          make(T, dims))),
          :(rand(                           T::Type{<:$Cont}, dims::Dims) =                 rand(GLOBAL_RNG, make(T, dims))),
          :(rand(r::AbstractRNG,            T::Type{<:$Cont}, dims::Integer...) =           rand(r,          make(T, Dims(dims)))),
          :(rand(                           T::Type{<:$Cont}, dims::Integer...) =           rand(GLOBAL_RNG, make(T, Dims(dims)))),

          :(rand(r::AbstractRNG, X,         T::Type{<:$Cont}, dims::Dims) =                 rand(r,          make(T, X, dims))),
          :(rand(                X,         T::Type{<:$Cont}, dims::Dims) =                 rand(GLOBAL_RNG, make(T, X, dims))),
          :(rand(r::AbstractRNG, X,         T::Type{<:$Cont}, dims::Integer...) =           rand(r,          make(T, X, Dims(dims)))),
          :(rand(                X,         T::Type{<:$Cont}, dims::Integer...) =           rand(GLOBAL_RNG, make(T, X, Dims(dims)))),

          :(rand(r::AbstractRNG, ::Type{X}, T::Type{<:$Cont}, dims::Dims)       where {X} = rand(r,          make(T, X, dims))),
          :(rand(                ::Type{X}, T::Type{<:$Cont}, dims::Dims)       where {X} = rand(GLOBAL_RNG, make(T, X, dims))),
          :(rand(r::AbstractRNG, ::Type{X}, T::Type{<:$Cont}, dims::Integer...) where {X} = rand(r,          make(T, X, Dims(dims)))),
          :(rand(                ::Type{X}, T::Type{<:$Cont}, dims::Integer...) where {X} = rand(GLOBAL_RNG, make(T, X, Dims(dims)))),
        ]
    esc(Expr(:block, definitions...))
end

@make_array_container(Array)
@make_array_container(BitArray)


## dicts

# again same inference bug
# TODO: extend to AbstractDict ? (needs to work-around the inderence bug)
default_sampling(::Type{<:Dict{K,V}}) where {K,V} = Pair{K,V}
default_sampling(D::Type{<:Dict}) = throw(ArgumentError("under-specified scalar type for $D"))

rand!(A::AbstractDict{K,V}, dist::Union{Type{<:Pair},Distribution{<:Pair}}=make(Pair, K, V)) where {K,V} =
    rand!(GLOBAL_RNG, A, dist)

rand!(rng::AbstractRNG, A::AbstractDict{K,V},
      dist::Union{Type{<:Pair},Distribution{<:Pair}}=make(Pair, K, V)) where {K,V} =
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
rand!(A::AbstractSet{T}, ::Type{X}=default_sampling(A)) where {T,X} = rand!(GLOBAL_RNG, A, X)

rand!(rng::AbstractRNG, A::AbstractSet, X) = rand!(rng, A, Sampler(rng, X))
rand!(rng::AbstractRNG, A::AbstractSet{T}, ::Type{X}=default_sampling(A)) where {T,X} = rand!(rng, A, Sampler(rng, X))

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

rand(rng::AbstractRNG, chars, ::Type{String}, n::Integer=8) = rand(rng, make(String, chars, n))
rand(                  chars, ::Type{String}, n::Integer=8) = rand(GLOBAL_RNG, make(String, chars, n))

rand(rng::AbstractRNG, ::Type{String}, n::Integer=8) = rand(rng, make(String, n))
rand(                  ::Type{String}, n::Integer=8) = rand(GLOBAL_RNG, make(String, n))


## NTuple as a container

rand(r::AbstractRNG, X,         ::Type{NTuple{N}})   where {N}   = rand(r,          make(NTuple{N}, X))
rand(                X,         ::Type{NTuple{N}})   where {N}   = rand(GLOBAL_RNG, make(NTuple{N}, X))
rand(r::AbstractRNG, ::Type{X}, ::Type{NTuple{N}})   where {X,N} = rand(r,          make(NTuple{N}, X))
rand(                ::Type{X}, ::Type{NTuple{N}})   where {X,N} = rand(GLOBAL_RNG, make(NTuple{N}, X))
rand(r::AbstractRNG,            ::Type{NTuple{N,X}}) where {X,N} = rand(r,          make(NTuple{N}, X))
rand(                           ::Type{NTuple{N,X}}) where {X,N} = rand(GLOBAL_RNG, make(NTuple{N}, X))

### disambiguate

rand(::AbstractRNG, ::Type{Tuple{}}) = ()
rand(               ::Type{Tuple{}}) = ()
