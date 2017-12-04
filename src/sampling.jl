# definition of samplers and random generation


## Uniform

Sampler(rng::AbstractRNG, d::Union{UniformWrap,UniformType}, n::Repetition) =
    Sampler(rng, d[], n)


## floats

### override def from Random

Sampler(rng::AbstractRNG, ::Type{T}, n::Repetition) where {T<:AbstractFloat} =
    Sampler(rng, CloseOpen01(T), n)

### fall-back on Random definitions
rand(r::AbstractRNG, ::SamplerTrivial{CloseOpen01{T}}) where {T} =
    rand(r, SamplerTrivial(Random.CloseOpen01{T}()))

rand(r::AbstractRNG, ::SamplerTrivial{CloseOpen12{T}}) where {T} =
    rand(r, SamplerTrivial(Random.CloseOpen12{T}()))

### CloseOpenAB

Sampler(rng::AbstractRNG, d::CloseOpenAB{T}, n::Repetition) where {T} =
    SamplerTag{CloseOpenAB{T}}((a=d.a, d=d.b - d.a, sp=Sampler(rng, CloseOpen01{T}(), n)))

rand(rng::AbstractRNG, sp::SamplerTag{CloseOpenAB{T}}) where {T} =
    sp.data.a + sp.data.d  * rand(rng, sp.data.sp)


## sampler for pairs and complex numbers

function Sampler(rng::AbstractRNG, u::Combine2{T}, n::Repetition) where T <: Union{Pair,Complex}
    sp1 = Sampler(rng, u.x, n)
    sp2 = u.x == u.y ? sp1 : Sampler(rng, u.y, n)
    SamplerTag{Cont{T}}((sp1, sp2))
end

rand(rng::AbstractRNG, sp::SamplerTag{Cont{T}}) where {T<:Union{Pair,Complex}} =
    T(rand(rng, sp.data[1]), rand(rng, sp.data[2]))


### additional methods for complex numbers

Sampler(rng::AbstractRNG, u::Combine1{Complex}, n::Repetition) =
    Sampler(rng, Combine(Complex, u.x, u.x), n)

Sampler(rng::AbstractRNG, ::Type{Complex{T}}, n::Repetition) where {T<:Real} =
    Sampler(rng, Combine(Complex, T, T), n)


## Normal & Exponential

rand(rng::AbstractRNG, ::SamplerTrivial{Normal01{T}}) where {T<:Union{AbstractFloat,Complex{<:AbstractFloat}}} =
    randn(rng, T)

Sampler(rng::AbstractRNG, d::Normalμσ{T}, n::Repetition) where {T} =
    SamplerSimple(d, Sampler(rng, Normal(T), n))

rand(rng::AbstractRNG, sp::SamplerSimple{Normalμσ{T},<:Sampler}) where {T} =
    sp[].μ + sp[].σ  * rand(rng, sp.data)

rand(rng::AbstractRNG, ::SamplerTrivial{Exponential1{T}}) where {T<:AbstractFloat} =
    randexp(rng, T)

Sampler(rng::AbstractRNG, d::Exponentialθ{T}, n::Repetition) where {T} =
    SamplerSimple(d, Sampler(rng, Exponential(T), n))

rand(rng::AbstractRNG, sp::SamplerSimple{Exponentialθ{T},<:Sampler}) where {T} =
    sp[].θ * rand(rng, sp.data)
