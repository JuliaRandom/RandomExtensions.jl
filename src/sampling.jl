# definition of samplers and random generation


# allows to call `Sampler` only when the the arg isn't a Sampler itself
sampler(RNG::Type{<:AbstractRNG}, X,          n::Repetition=Val(Inf))           = Sampler(RNG, X, n)
sampler(RNG::Type{<:AbstractRNG}, ::Type{X},  n::Repetition=Val(Inf)) where {X} = Sampler(RNG, X, n)
sampler(RNG::Type{<:AbstractRNG}, X::Sampler, n::Repetition=Val(Inf))           = X

sampler(rng::AbstractRNG, X, n::Repetition=Val(Inf)) = sampler(typeof(rng), X, n)

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

### new intervals 01

# TODO: optimize for BigFloat

for CO = (:OpenClose01, :OpenOpen01, :CloseClose01)
    @eval Sampler(RNG::Type{<:AbstractRNG}, I::$CO{T}, n::Repetition) where {T} =
              SamplerSimple(I, CloseOpen01(T))
end

rand(r::AbstractRNG, sp::SamplerSimple{OpenClose01{T}}) where {T} =
    one(T) - rand(r, sp.data)

rand(r::AbstractRNG, sp::SamplerSimple{OpenOpen01{T}}) where {T} =
    while true
        x = rand(r, sp.data)
        x != zero(T) && return x
    end

rand(r::AbstractRNG, sp::SamplerSimple{CloseClose01{T}}) where {T} =
    rand(r, sp.data) / prevfloat(one(T))

### CloseOpenAB

for (CO, CO01) = (CloseOpenAB => CloseOpen01,
                  OpenCloseAB => OpenClose01,
                  CloseCloseAB => CloseClose01,
                  OpenOpenAB => OpenOpen01)

    @eval Sampler(RNG::Type{<:AbstractRNG}, d::$CO{T}, n::Repetition) where {T} =
        SamplerTag{$CO{T}}((a=d.a, d=d.b - d.a, sp=Sampler(RNG, $CO01{T}(), n)))

    @eval rand(rng::AbstractRNG, sp::SamplerTag{$CO{T}}) where {T} =
        sp.data.a + sp.data.d  * rand(rng, sp.data.sp)
end

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

find_type(::Type{Pair},              x, y)             = Pair{val_gentype(x), val_gentype(y)}
find_type(::Type{Pair{X}},           _, y) where {X}   = Pair{X, val_gentype(y)}
find_type(::Type{Pair{X,Y} where X}, x, _) where {Y}   = Pair{val_gentype(x), Y}
find_type(::Type{Pair{X,Y}},         _, _) where {X,Y} = Pair{X,Y}

find_type(::Type{Complex},    x) = Complex{val_gentype(x)}
find_type(T::Type{<:Complex}, _) = T

find_type(::Type{Complex},    x, y) = Complex{promote_type(val_gentype(x), val_gentype(y))}
find_type(T::Type{<:Complex}, _, _) = T

function Sampler(RNG::Type{<:AbstractRNG}, u::Make2{T}, n::Repetition) where T <: Union{Pair,Complex}
    sp1 = sampler(RNG, u.x, n)
    sp2 = u.x == u.y ? sp1 : sampler(RNG, u.y, n)
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
    d = Dict{DataType,Int}()
    sps = []
    for t in T.parameters
        i = get(d, t, nothing)
        if i === nothing
            push!(sps, :(Sampler(RNG, $t, n)))
            d[t] = length(sps)
        else
            push!(sps, Val(i))
        end
    end
    :(SamplerTag{Cont{T}}(tuple($(sps...))))
end

@generated function rand(rng::AbstractRNG, sp::SamplerTag{Cont{T},S}) where {T<:Tuple,S<:Tuple}
    @assert fieldcount(T) == fieldcount(S)
    rands = []
    for i = 1:fieldcount(T)
        j = fieldtype(S, i) <: Val ?
              fieldtype(S, i).parameters[1] :
              i
        push!(rands, :(convert($(fieldtype(T, i)),
                               rand(rng, sp.data[$j]))))
    end
    :(tuple($(rands...)))
end

#### with make

# implement make(Tuple, S1, S2...), e.g. for rand(make(Tuple, Int, 1:3)),
# and       make(NTuple{N}, S)

@generated function _make(::Type{T}, args...) where T <: Tuple
    if isempty(args)
        TT = T === Tuple ? Tuple{} :
            T === NTuple ? Tuple{} :
            T isa UnionAll && Type{T} <: Type{NTuple{N}} where N ? T{default_sampling(Tuple)} :
            T
        return :(Make0{$TT}())
    end
    isNT = length(args) == 1 && T !== Tuple && (
        T <: NTuple || !isa(T, UnionAll)) # !isa(Tuple, UnionAll) !!

    types = [t <: Type ? t.parameters[1] : gentype(t) for t in args]
    TT = T === Tuple ? Tuple{types...} :
        isNT ? (T isa UnionAll ? Tuple{fill(types[1], fieldcount(T))...} : T ) :
        T
    samples = [t <: Type ? :(UniformType{$(t.parameters[1])}()) :
               :(args[$i]) for (i, t) in enumerate(args)]
    if isNT
        :(Make1{$TT}($(samples[1])))
    else
        quote
            if T !== Tuple && fieldcount(T) != length(args)
                throw(ArgumentError("wrong number of provided argument with $T (should be $(fieldcount(T)))"))
            else
                Make1{$TT}(tuple($(samples...)))
            end
        end
    end
end

make(T::Type{<:Tuple}, args...) = _make(T, args...)

# make(Tuple, X, n::Integer)

default_sampling(::Type{Tuple}) = Float64

make(::Type{Tuple}, X,         n::Integer)           = make(NTuple{Int(n)}, X)
make(::Type{Tuple}, ::Type{X}, n::Integer) where {X} = make(NTuple{Int(n)}, X)

make(::Type{Tuple}, n::Integer) = make(Tuple, default_sampling(Tuple), Int(n))

# NTuple{N,T} where N
make(::Type{NTuple{N,T} where N},            n::Integer) where {T}   = make(NTuple{Int(n),T})
make(::Type{NTuple{N,T} where N}, X,         n::Integer) where {T}   = make(NTuple{Int(n),T}, X)
make(::Type{NTuple{N,T} where N}, ::Type{X}, n::Integer) where {T,X} = make(NTuple{Int(n),T}, X)

# disambiguate

make(::Type{T}, X)         where {T<:Tuple}   = _make(T, X)
make(::Type{T}, ::Type{X}) where {T<:Tuple,X} = _make(T, X)

make(::Type{T}, X,         Y)         where {T<:Tuple}     = _make(T, X, Y)
make(::Type{T}, ::Type{X}, Y)         where {T<:Tuple,X}   = _make(T, X, Y)
make(::Type{T}, X,         ::Type{Y}) where {T<:Tuple,Y}   = _make(T, X, Y)
make(::Type{T}, ::Type{X}, ::Type{Y}) where {T<:Tuple,X,Y} = _make(T, X, Y)

make(::Type{T}, X,         Y,         Z)         where {T<:Tuple}       = _make(T, X, Y, Z)
make(::Type{T}, ::Type{X}, Y,         Z)         where {T<:Tuple,X}     = _make(T, X, Y, Z)
make(::Type{T}, X,         ::Type{Y}, Z)         where {T<:Tuple,Y}     = _make(T, X, Y, Z)
make(::Type{T}, ::Type{X}, ::Type{Y}, Z)         where {T<:Tuple,X,Y}   = _make(T, X, Y, Z)
make(::Type{T}, X,         Y,         ::Type{Z}) where {T<:Tuple,Z}     = _make(T, X, Y, Z)
make(::Type{T}, ::Type{X}, Y,         ::Type{Z}) where {T<:Tuple,X,Z}   = _make(T, X, Y, Z)
make(::Type{T}, X,         ::Type{Y}, ::Type{Z}) where {T<:Tuple,Y,Z}   = _make(T, X, Y, Z)
make(::Type{T}, ::Type{X}, ::Type{Y}, ::Type{Z}) where {T<:Tuple,X,Y,Z} = _make(T, X, Y, Z)

##### Sampler for general tuples (rand is already implemented above, like for rand(Tuple{...})

@generated function Sampler(RNG::Type{<:AbstractRNG}, c::Make1{T,X}, n::Repetition) where {T<:Tuple,X<:Tuple}
    @assert fieldcount(T) == fieldcount(X)
    sps = [:(sampler(RNG, c.x[$i], n)) for i in 1:length(T.parameters)]
    :(SamplerTag{Cont{T}}(tuple($(sps...))))
end

Sampler(RNG::Type{<:AbstractRNG}, ::Make0{T}, n::Repetition) where {T<:Tuple} =
    Sampler(RNG, T, n)

##### for "NTuple-like"

# should catch Tuple{Integer,Integer} which is not NTuple, or even Tuple{Int,UInt}, when only one sampler was passed

Sampler(RNG::Type{<:AbstractRNG}, c::Make1{T,X}, n::Repetition) where {T<:Tuple,X} =
    SamplerTag{Cont{T}}(sampler(RNG, c.x, n))

@generated function rand(rng::AbstractRNG, sp::SamplerTag{Cont{T},S}) where {T<:Tuple,S<:Sampler}
    rands = [:(convert($(T.parameters[i]), rand(rng, sp.data))) for i in 1:fieldcount(T)]
    :(tuple($(rands...)))
end


## collections

### sets/dicts

const SetDict = Union{AbstractSet,AbstractDict}

make(T::Type{<:SetDict}, X,         n::Integer)           = Make2{find_type(T, X, n)}(X , Int(n))
make(T::Type{<:SetDict}, ::Type{X}, n::Integer) where {X} = Make2{find_type(T, X, n)}(X , Int(n))
make(T::Type{<:SetDict},            n::Integer)           = make(T, default_sampling(T), Int(n))

Sampler(RNG::Type{<:AbstractRNG}, c::Make2{T}, n::Repetition) where {T<:SetDict} =
    SamplerTag{Cont{T}}((sampler(RNG, c.x, n), c.y))

function rand(rng::AbstractRNG, sp::SamplerTag{Cont{S}}) where {S<:SetDict}
    # assuming S() creates an empty set/dict
    s = sizehint!(S(), sp.data[2])
    _rand!(rng, s, sp.data[2], sp.data[1])
end

### sets

default_sampling(::Type{<:AbstractSet}) = Float64
default_sampling(::Type{<:AbstractSet{T}}) where {T} = T

#### Set

find_type(::Type{Set},    X, _)           = Set{val_gentype(X)}
find_type(::Type{Set{T}}, _, _) where {T} = Set{T}

### BitSet

default_sampling(::Type{BitSet}) = Int8 # almost arbitrary, may change

find_type(::Type{BitSet}, _, _) = BitSet


### dicts

# again same inference bug
# TODO: extend to AbstractDict ? (needs to work-around the inderence bug)
default_sampling(::Type{Dict{K,V}}) where {K,V} = Pair{K,V}
default_sampling(D::Type{<:Dict})               = throw(ArgumentError("under-specified scalar type for $D"))

find_type(D::Type{<:AbstractDict{K,V}}, _,      ::Integer) where {K,V} = D
find_type(D::Type{<:AbstractDict{K,V}}, ::Type, ::Integer) where {K,V} = D

#### Dict

find_type(::Type{Dict{K}},           X, ::Integer) where {K} = Dict{K,fieldtype(val_gentype(X), 2)}
find_type(::Type{Dict{K,V} where K}, X, ::Integer) where {V} = Dict{fieldtype(val_gentype(X), 1),V}
find_type(::Type{Dict},              X, ::Integer)           = Dict{fieldtype(val_gentype(X), 1),fieldtype(val_gentype(X), 2)}


### AbstractArray

default_sampling(::Type{<:AbstractArray{T}}) where {T} = T
default_sampling(::Type{<:AbstractArray})              = Float64

make(A::Type{<:AbstractArray}, X,         dims::Integer...)           = make(A, X, Dims(dims))
make(A::Type{<:AbstractArray}, ::Type{X}, dims::Integer...) where {X} = make(A, X, Dims(dims))

make(A::Type{<:AbstractArray}, dims::Dims)       = make(A, default_sampling(A), dims)
make(A::Type{<:AbstractArray}, dims::Integer...) = make(A, default_sampling(A), Dims(dims))


Sampler(RNG::Type{<:AbstractRNG}, c::Make2{A}, n::Repetition) where {A<:AbstractArray} =
    SamplerTag{A}((sampler(RNG, c.x, n), c.y))

rand(rng::AbstractRNG, sp::SamplerTag{A}) where {A<:AbstractArray} =
    rand!(rng, A(undef, sp.data[2]), sp.data[1])


#### Array

val_gentype(X)                   = gentype(X)
val_gentype(::Type{X}) where {X} = X

# cf. inference bug https://github.com/JuliaLang/julia/issues/28762
# we have to write out all combinations for getting proper inference
find_type(A::Type{Array{T}},           _, ::Dims{N}) where {T, N} = Array{T, N}
find_type(A::Type{Array{T,N}},         _, ::Dims{N}) where {T, N} = Array{T, N}
find_type(A::Type{Array{T,N} where T}, X, ::Dims{N}) where {N}    = Array{val_gentype(X), N}
find_type(A::Type{Array},              X, ::Dims{N}) where {N}    = Array{val_gentype(X), N}


#### BitArray

default_sampling(::Type{<:BitArray}) = Bool

find_type(::Type{BitArray{N}}, _, ::Dims{N}) where {N} = BitArray{N}
find_type(::Type{BitArray},    _, ::Dims{N}) where {N} = BitArray{N}


### sparse vectors & matrices

make(p::AbstractFloat, X, dims::Dims{1}) = Make3{SparseVector{   val_gentype(X), Int}}(X, dims, p)
make(p::AbstractFloat, X, dims::Dims{2}) = Make3{SparseMatrixCSC{val_gentype(X), Int}}(X, dims, p)

make(p::AbstractFloat, X, dims::Integer...) = make(p, X, Dims(dims))
make(p::AbstractFloat, dims::Dims)          = make(p, Float64, dims)
make(p::AbstractFloat, dims::Integer...)    = make(p, Float64, Dims(dims))

Sampler(RNG::Type{<:AbstractRNG}, c::Make3{A}, n::Repetition) where {A<:AbstractSparseArray} =
    SamplerTag{A}((sampler(RNG, c.x, n), c.y, c.z))

rand(rng::AbstractRNG, sp::SamplerTag{A}) where {A<:SparseVector} =
    sprand(rng, sp.data[2][1], sp.data[3], (r, n)->rand(r, sp.data[1], n))

rand(rng::AbstractRNG, sp::SamplerTag{A}) where {A<:SparseMatrixCSC} =
    sprand(rng, sp.data[2][1], sp.data[2][2], sp.data[3], (r, n)->rand(r, sp.data[1], n), gentype(sp.data[1]))


### String as a scalar

let b = UInt8['0':'9';'A':'Z';'a':'z'],
    s = Sampler(MersenneTwister, b, Val(Inf)) # cache for the likely most common case

    global Sampler, rand, make

    make(::Type{String})                                   = Make2{String}(8, b)
    make(::Type{String}, chars)                            = Make2{String}(8, chars)
    make(::Type{String}, ::Type{C}) where C                = Make2{String}(8, C)
    make(::Type{String}, n::Integer)                       = Make2{String}(Int(n), b)
    make(::Type{String}, chars,      n::Integer)           = Make2{String}(Int(n), chars)
    make(::Type{String}, ::Type{C},  n::Integer) where {C} = Make2{String}(Int(n), C)
    make(::Type{String}, n::Integer, chars)                = Make2{String}(Int(n), chars)
    make(::Type{String}, n::Integer, ::Type{C}) where {C}  = Make2{String}(Int(n), C)

    Sampler(RNG::Type{<:AbstractRNG}, ::Type{String}, n::Repetition) =
        SamplerTag{Cont{String}}((RNG === MersenneTwister ? s : Sampler(RNG, b, n)) => 8)

    function Sampler(RNG::Type{<:AbstractRNG}, c::Make2{String}, n::Repetition)
        sp = RNG === MersenneTwister && c.y === b ?
            s : sampler(RNG, c.y, n)
        SamplerTag{Cont{String}}(sp => c.x)
    end

    rand(rng::AbstractRNG, sp::SamplerTag{Cont{String}}) = String(rand(rng, sp.data.first, sp.data.second))
end
