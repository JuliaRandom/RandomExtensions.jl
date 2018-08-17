# definition of samplers and random generation


## Uniform

Sampler(RNG::Type{<:AbstractRNG}, d::Union{UniformWrap,UniformType}, n::Repetition) =
    Sampler(RNG, d[], n)


## floats

### override def from Random

Sampler(RNG::Type{<:AbstractRNG}, ::Type{T}, n::Repetition) where {T<:AbstractFloat} =
    Sampler(RNG, CloseOpen01(T), n)

### fall-back on Random definitions
rand(r::AbstractRNG, ::SamplerTrivial{CloseOpen01{T}}) where {T} =
    rand(r, SamplerTrivial(Random.CloseOpen01{T}()))

rand(r::AbstractRNG, ::SamplerTrivial{CloseOpen12{T}}) where {T} =
    rand(r, SamplerTrivial(Random.CloseOpen12{T}()))

### CloseOpenAB

Sampler(RNG::Type{<:AbstractRNG}, d::CloseOpenAB{T}, n::Repetition) where {T} =
    SamplerTag{CloseOpenAB{T}}((a=d.a, d=d.b - d.a, sp=Sampler(RNG, CloseOpen01{T}(), n)))

rand(rng::AbstractRNG, sp::SamplerTag{CloseOpenAB{T}}) where {T} =
    sp.data.a + sp.data.d  * rand(rng, sp.data.sp)


## sampler for pairs and complex numbers

function Sampler(RNG::Type{<:AbstractRNG}, u::Combine2{T}, n::Repetition) where T <: Union{Pair,Complex}
    sp1 = Sampler(RNG, u.x, n)
    sp2 = u.x == u.y ? sp1 : Sampler(RNG, u.y, n)
    SamplerTag{Cont{T}}((sp1, sp2))
end

rand(rng::AbstractRNG, sp::SamplerTag{Cont{T}}) where {T<:Union{Pair,Complex}} =
    T(rand(rng, sp.data[1]), rand(rng, sp.data[2]))


### additional convenience methods

# rand(Pair{A,B}) => rand(Combine(Pair{A,B}, A, B))
Sampler(RNG::Type{<:AbstractRNG}, ::Type{Pair{A,B}}, n::Repetition) where {A,B} =
    Sampler(RNG, Combine(Pair{A,B}, A, B), n)

# rand(Combine(Complex, x)) => rand(Combine(Combine, x, x))
Sampler(RNG::Type{<:AbstractRNG}, u::Combine1{T}, n::Repetition) where {T<:Complex} =
    Sampler(RNG, Combine(T, u.x, u.x), n)

# rand(Complex{T}) => rand(Combine(Complex{T}, T, T)) (redundant with implem in Random)
Sampler(RNG::Type{<:AbstractRNG}, ::Type{Complex{T}}, n::Repetition) where {T<:Real} =
    Sampler(RNG, Combine(Complex{T}, T, T), n)


## Normal & Exponential

rand(rng::AbstractRNG, ::SamplerTrivial{Normal01{T}}) where {T<:Union{AbstractFloat,Complex{<:AbstractFloat}}} =
    randn(rng, T)

Sampler(RNG::Type{<:AbstractRNG}, d::Normalμσ{T}, n::Repetition) where {T} =
    SamplerSimple(d, Sampler(RNG, Normal(T), n))

rand(rng::AbstractRNG, sp::SamplerSimple{Normalμσ{T},<:Sampler}) where {T} =
    sp[].μ + sp[].σ  * rand(rng, sp.data)

rand(rng::AbstractRNG, ::SamplerTrivial{Exponential1{T}}) where {T<:AbstractFloat} =
    randexp(rng, T)

Sampler(RNG::Type{<:AbstractRNG}, d::Exponentialθ{T}, n::Repetition) where {T} =
    SamplerSimple(d, Sampler(RNG, Exponential(T), n))

rand(rng::AbstractRNG, sp::SamplerSimple{Exponentialθ{T},<:Sampler}) where {T} =
    sp[].θ * rand(rng, sp.data)


## random elements from pairs

Sampler(RNG::Type{<:AbstractRNG}, t::Pair, n::Repetition) =
    SamplerSimple(t, Sampler(RNG, Bool, n))

rand(rng::AbstractRNG, sp::SamplerSimple{<:Pair}) =
    @inbounds return sp[][1 + rand(rng, sp.data)]
