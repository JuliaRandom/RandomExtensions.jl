# iterating over on-the-fly generated random values

## Rand

struct Rand{R<:AbstractRNG,S<:Sampler}
    rng::R
    sp::S
end

# X can be an explicit Distribution, or an implicit one like 1:10
Rand(rng::AbstractRNG, X) = Rand(rng, Sampler(rng, X))
Rand(rng::AbstractRNG, ::Type{X}=Float64) where {X} = Rand(rng, Sampler(rng, X))

Rand(X) = Rand(GLOBAL_RNG, X)
Rand(::Type{X}=Float64) where {X} = Rand(GLOBAL_RNG, X)

(R::Rand)(args...) = rand(R.rng, R.sp, args...)

Base.iterate(iter::Union{Rand,Distribution}, R::Rand=iter) = R(), R
Base.IteratorSize(::Type{<:Rand}) = Base.IsInfinite()

Base.IteratorEltype(::Type{<:Rand}) = Base.HasEltype()
Base.eltype(::Type{<:Rand{R, <:Sampler{T}}}) where {R,T} = T

# convenience iteration over distributions

Base.iterate(d::Distribution) = iterate(Rand(GLOBAL_RNG, d))
Base.IteratorSize(::Type{<:Distribution}) = Base.IsInfinite()
