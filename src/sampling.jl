# definition of samplers and random generation


## Uniform

Sampler(RNG::Type{<:AbstractRNG}, d::Union{UniformWrap,UniformType}, n::Repetition) =
    Sampler(RNG, d[], n)


## floats

### override def from Random

Sampler(RNG::Type{<:AbstractRNG}, ::Type{T}, n::Repetition) where {T<:AbstractFloat} =
    Sampler(RNG, CloseOpen01(T), n)

for CO in (:CloseOpen01, :CloseOpen12)
    @eval Sampler(::Type{<:AbstractRNG}, I::$CO{BigFloat}, ::Repetition) =
        Random.SamplerBigFloat{Random.$CO{BigFloat}}(precision(BigFloat))
end

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


## Bernoulli

Sampler(RNG::Type{<:AbstractRNG}, b::Bernoulli, n::Repetition) =
    SamplerTag{typeof(b)}(b.p+1.0)

rand(rng::AbstractRNG, sp::SamplerTag{Bernoulli{T}}) where {T} =
    ifelse(rand(rng, CloseOpen12()) < sp.data, one(T), zero(T))


## random elements from pairs

Sampler(RNG::Type{<:AbstractRNG}, t::Pair, n::Repetition) =
    SamplerSimple(t, Sampler(RNG, Bool, n))

rand(rng::AbstractRNG, sp::SamplerSimple{<:Pair}) =
    @inbounds return sp[][1 + rand(rng, sp.data)]


## composite types

### sampler for pairs and complex numbers

find_type(T::Type{<:Union{Pair,Complex}}, args...) = find_deduced_type(T, args...)

function Sampler(RNG::Type{<:AbstractRNG}, u::Make2{T}, n::Repetition) where T <: Union{Pair,Complex}
    sp1 = Sampler(RNG, u.x, n)
    sp2 = u.x == u.y ? sp1 : Sampler(RNG, u.y, n)
    SamplerTag{Cont{T}}((sp1, sp2))
end

rand(rng::AbstractRNG, sp::SamplerTag{Cont{T}}) where {T<:Union{Pair,Complex}} =
    T(rand(rng, sp.data[1]), rand(rng, sp.data[2]))


#### additional convenience methods

# rand(Pair{A,B}) => rand(make(Pair{A,B}, A, B))
Sampler(RNG::Type{<:AbstractRNG}, ::Type{Pair{A,B}}, n::Repetition) where {A,B} =
    Sampler(RNG, make(Pair{A,B}, A, B), n)

# rand(make(Complex, x)) => rand(make(Complex, x, x))
Sampler(RNG::Type{<:AbstractRNG}, u::Make1{T}, n::Repetition) where {T<:Complex} =
    Sampler(RNG, make(T, u.x, u.x), n)

# rand(Complex{T}) => rand(make(Complex{T}, T, T)) (redundant with implem in Random)
Sampler(RNG::Type{<:AbstractRNG}, ::Type{Complex{T}}, n::Repetition) where {T<:Real} =
    Sampler(RNG, make(Complex{T}, T, T), n)


### sampler for tuples

@generated function Sampler(RNG::Type{<:AbstractRNG}, ::Type{T}, n::Repetition) where {T<:Tuple}
    U = unique(T.parameters)
    sps = [:(Sampler(RNG, $(U[i]), n)) for i in 1:length(U)]
    :(SamplerTag{Cont{T}}(tuple($(sps...))))
end

@generated function rand(rng::AbstractRNG, sp::SamplerTag{Cont{T},S}) where {T<:Tuple,S<:Tuple}
    rands = []
    for i = 1:fieldcount(T)
        # if as many fields for S and T, don't try to shortcut, as it's
        # unnecessary, and even wrong when sp was created from Make
        k = fieldcount(S) == fieldcount(T) ? i : 1
        for j = k:i
            if fieldtype(T, i) == gentype(fieldtype(S, j))
                push!(rands, :(rand(rng, sp.data[$j])))
                break
            end
        end
    end
    :(tuple($(rands...)))
end

#### with make

# implement make(Tuple, S1, S2...), e.g. for rand(make(Tuple, Int, 1:3)),
# and       make(NTuple{N}, S)

@generated function _make(::Type{Tuple}, args...)
    types = [t <: Type ? t.parameters[1] : gentype(t) for t in args]
    T = Tuple{types...}
    samples = [t <: Type ? :(UniformType{$(t.parameters[1])}()) :
               :(args[$i]) for (i, t) in enumerate(args)]
    :(Make1{$T}(tuple($(samples...))))
end

make(::Type{Tuple}, args...) = _make(Tuple, args...)

@generated function _make(::Type{NTuple{N}}, arg) where {N}
    T, a = arg <: Type ?
        (arg.parameters[1], :(Uniform(arg))) :
        (gentype(arg), :arg)
    :(Make1{NTuple{N,$T}}($a))
end

make(::Type{NTuple{N}}, X) where {N} = _make(NTuple{N}, X)
make(::Type{NTuple{N}}, ::Type{X}) where {N,X} = _make(NTuple{N}, X)

# disambiguate

make(::Type{Tuple}, X) = _make(Tuple, X)
make(::Type{Tuple}, ::Type{X}) where {X} = _make(Tuple, X)

make(::Type{Tuple}, X, Y) = _make(Tuple, X, Y)
make(::Type{Tuple}, ::Type{X}, Y) where {X} = _make(Tuple, X, Y)
make(::Type{Tuple}, X, ::Type{Y}) where {Y} = _make(Tuple, X, Y)
make(::Type{Tuple}, ::Type{X}, ::Type{Y}) where {X,Y} = _make(Tuple, X, Y)


# Sampler (rand is already implemented above, like for rand(Tuple{...})

@generated function Sampler(RNG::Type{<:AbstractRNG}, c::Make1{T,X}, n::Repetition) where {T<:Tuple,X<:Tuple}
    sps = [:(Sampler(RNG, c.x[$i], n)) for i in 1:length(T.parameters)]
    :(SamplerTag{Cont{T}}(tuple($(sps...))))
end

Sampler(RNG::Type{<:AbstractRNG}, c::Make1{T,X}, n::Repetition) where {T<:Tuple,X} =
    SamplerTag{Cont{T}}(Sampler(RNG, c.x, n))

@generated function rand(rng::AbstractRNG, sp::SamplerTag{Cont{T},S}) where {T<:NTuple,S<:Sampler}
    rands = fill(:(rand(rng, sp.data)), fieldcount(T))
    :(tuple($(rands...)))
end


## collections

### BitSet

default_sampling(::Type{BitSet}) = Int8 # almost arbitrary, may change

make(::Type{BitSet},            n::Integer)           = Make2{BitSet}(default_sampling(BitSet), Int(n))
make(::Type{BitSet}, X,         n::Integer)           = Make2{BitSet}(X, Int(n))
make(::Type{BitSet}, ::Type{X}, n::Integer) where {X} = Make2{BitSet}(X, Int(n))

Sampler(RNG::Type{<:AbstractRNG}, c::Make{BitSet}, n::Repetition) =
    SamplerTag{BitSet}((Sampler(RNG, c.x, n), c.y))

function rand(rng::AbstractRNG, sp::SamplerTag{BitSet})
    s = sizehint!(BitSet(), sp.data[2])
    _rand!(rng, s, sp.data[2], sp.data[1])
end


### BitArray

default_sampling(::Type{<:BitArray}) = Bool

make(::Type{BitArray{N}}, X,         dims::Dims{N}) where {N}   = Make2{BitArray{N}}(X, dims)
make(::Type{BitArray{N}}, ::Type{X}, dims::Dims{N}) where {N,X} = Make2{BitArray{N}}(X, dims)
make(::Type{BitArray},    X,         dims::Dims{N}) where {N}   = Make2{BitArray{N}}(X, dims)
make(::Type{BitArray},    ::Type{X}, dims::Dims{N}) where {N,X} = Make2{BitArray{N}}(X, dims)

make(B::Type{<:BitArray},         X, dims::Integer...)           = make(B, X, Dims(dims))
make(B::Type{<:BitArray}, ::Type{X}, dims::Integer...) where {X} = make(B, X, Dims(dims))

make(B::Type{<:BitArray}, dims::Dims)       = make(B, default_sampling(B), dims)
make(B::Type{<:BitArray}, dims::Integer...) = make(B, default_sampling(B), Dims(dims))

Sampler(RNG::Type{<:AbstractRNG}, c::Make{B}, n::Repetition) where {B<:BitArray} =
    SamplerTag{B}((Sampler(RNG, c.x, n), c.y))

rand(rng::AbstractRNG, sp::SamplerTag{<:BitArray}) = rand!(rng, BitArray(undef, sp.data[2]), sp.data[1])


### String as a scalar

let b = UInt8['0':'9';'A':'Z';'a':'z'],
    s = Sampler(MersenneTwister, b, Val(Inf)) # cache for the likely most common case

    global Sampler, rand, make

    make(::Type{String}) = Make2{String}(8, b)
    make(::Type{String}, chars) = Make2{String}(8, chars)
    make(::Type{String}, n::Integer) = Make2{String}(Int(n), b)
    make(::Type{String}, chars, n::Integer) = Make2{String}(Int(n), chars)
    make(::Type{String}, n::Integer, chars) = Make2{String}(Int(n), chars)

    Sampler(RNG::Type{<:AbstractRNG}, ::Type{String}, n::Repetition) =
        SamplerTag{Cont{String}}((RNG === MersenneTwister ? s : Sampler(RNG, b, n)) => 8)

    function Sampler(RNG::Type{<:AbstractRNG}, c::Make2{String}, n::Repetition)
        sp = RNG === MersenneTwister && c.y === b ?
            s : Sampler(RNG, c.y, n)
        SamplerTag{Cont{String}}(sp => c.x)
    end

    rand(rng::AbstractRNG, sp::SamplerTag{Cont{String}}) = String(rand(rng, sp.data.first, sp.data.second))
end
