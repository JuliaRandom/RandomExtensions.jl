# definition of samplers and random generation


# allows to call `Sampler` only when the the arg isn't a Sampler itself
sampler(RNG::Type{<:AbstractRNG}, X,          n::Repetition=Val(Inf))           = Sampler(RNG, X, n)
sampler(RNG::Type{<:AbstractRNG}, ::Type{X},  n::Repetition=Val(Inf)) where {X} = Sampler(RNG, X, n)
sampler(RNG::Type{<:AbstractRNG}, X::Sampler, n::Repetition=Val(Inf))           = X

sampler(rng::AbstractRNG, X, n::Repetition=Val(Inf)) = sampler(typeof(rng), X, n)


## defaults

### 0-arg

make() = make(Float64)

### type

find_type(::Type{X}) where{X} = X

Sampler(RNG::Type{<:AbstractRNG}, ::Make0{X}, n::Repetition) where {X} =
    Sampler(RNG, X, n)

### object

# like Make1
struct MakeWrap{T,X} <: Make{T}
    x::X
end

# make(::Type) is intercepted in distribution.jl
make(x) = MakeWrap{gentype(x),typeof(x)}(x)

Sampler(RNG::Type{<:AbstractRNG}, x::MakeWrap, n::Repetition) =
    Sampler(RNG, x.x, n)


## Uniform

Sampler(RNG::Type{<:AbstractRNG}, d::Union{UniformWrap,UniformType}, n::Repetition) =
    Sampler(RNG, d[], n)


## floats

### fall-back on Random definitions

for CO in (:CloseOpen01, :CloseOpen12)
    @eval begin
        Sampler(RNG::Type{<:AbstractRNG}, ::$CO{T}, n::Repetition) where {T} =
            Sampler(RNG, Random.$CO{T}(), n)

        Sampler(::Type{<:AbstractRNG}, ::$CO{BigFloat}, ::Repetition) =
            Random.SamplerBigFloat{Random.$CO{BigFloat}}(precision(BigFloat))
    end
end

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

# optimizations (TODO: optimize for BigFloat too)

rand(r::AbstractRNG, sp::SamplerSimple{OpenOpen01{Float64}}) =
    reinterpret(Float64, reinterpret(UInt64, rand(r, sp.data)) | 0x0000000000000001)

rand(r::AbstractRNG, sp::SamplerSimple{OpenOpen01{Float32}}) =
    reinterpret(Float32, reinterpret(UInt32, rand(r, sp.data)) | 0x00000001)

rand(r::AbstractRNG, sp::SamplerSimple{OpenOpen01{Float16}}) =
    reinterpret(Float16, reinterpret(UInt16, rand(r, sp.data)) | 0x0001)

# prevfloat(T(2)) - 1 for IEEEFloat
upper01(::Type{Float64}) = 0.9999999999999998
upper01(::Type{Float32}) = 0.9999999f0
upper01(::Type{Float16}) = Float16(0.999)
upper01(::Type{BigFloat}) = prevfloat(one(BigFloat))

rand(r::AbstractRNG, sp::SamplerSimple{CloseClose01{T}}) where {T} =
    rand(r, sp.data) / upper01(T)

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

rand(rng::AbstractRNG, ::SamplerTrivial{Normal01{T}}) where {T<:NormalTypes} =
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

#### "simple scalar" (non-make) version

Sampler(RNG::Type{<:AbstractRNG}, ::Type{T}, n::Repetition) where {T<:Union{Tuple,NamedTuple}} =
    Sampler(RNG, make(T), n)

#### make

# implement make(Tuple, S1, S2...), e.g. for rand(make(Tuple, Int, 1:3)),
# and       make(NTuple{N}, S)

_find_type(::Type{T}) where {T<:Tuple} =
    T === Tuple ?
        Tuple{} :
    T === NTuple ?
        Tuple{} :
    T isa UnionAll && Type{T} <: Type{NTuple{N}} where N ?
        T{default_gentype(Tuple)} :
        T

function _find_type(::Type{T}, args...) where T <: Tuple
    types = [t <: Type ? t.parameters[1] : gentype(t) for t in args]
    TT = T === Tuple ?
        Tuple{types...} :
    _isNTuple(T, args...) ?
        (T isa UnionAll ? Tuple{fill(types[1], fieldcount(T))...} : T ) :
        T
    TT
end

_isNTuple(::Type{T}, args...) where {T<:Tuple} =
    length(args) == 1 && T !== Tuple && (
        T <: NTuple || !isa(T, UnionAll)) # !isa(Tuple, UnionAll) !!

@generated function _make(::Type{T}, args...) where T <: Tuple
    isempty(args) && return :(Make0{$(_find_type(T))}())
    TT = _find_type(T, args...)
    samples = [t <: Type ? :(UniformType{$(t.parameters[1])}()) :
               :(args[$i]) for (i, t) in enumerate(args)]
    if _isNTuple(T, args...)
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

default_sampling(::Type{Tuple}) = Uniform(Float64)

make(::Type{Tuple}, X,         n::Integer)           = make(NTuple{Int(n)}, X)
make(::Type{Tuple}, ::Type{X}, n::Integer) where {X} = make(NTuple{Int(n)}, X)

make(::Type{Tuple}, n::Integer) = make(Tuple, default_sampling(Tuple), Int(n))

# NTuple{N,T} where N
make(::Type{NTuple{N,T} where N},            n::Integer) where {T}   = make(NTuple{Int(n),T})
make(::Type{NTuple{N,T} where N}, X,         n::Integer) where {T}   = make(NTuple{Int(n),T}, X)
make(::Type{NTuple{N,T} where N}, ::Type{X}, n::Integer) where {T,X} = make(NTuple{Int(n),T}, X)

# disambiguate

for Tupl = (Tuple, NamedTuple)
    @eval begin
        make(::Type{T}, X)         where {T<:$Tupl}   = _make(T, X)
        make(::Type{T}, ::Type{X}) where {T<:$Tupl,X} = _make(T, X)

        make(::Type{T}, X,         Y)         where {T<:$Tupl}     = _make(T, X, Y)
        make(::Type{T}, ::Type{X}, Y)         where {T<:$Tupl,X}   = _make(T, X, Y)
        make(::Type{T}, X,         ::Type{Y}) where {T<:$Tupl,Y}   = _make(T, X, Y)
        make(::Type{T}, ::Type{X}, ::Type{Y}) where {T<:$Tupl,X,Y} = _make(T, X, Y)

        make(::Type{T}, X,         Y,         Z)         where {T<:$Tupl}       = _make(T, X, Y, Z)
        make(::Type{T}, ::Type{X}, Y,         Z)         where {T<:$Tupl,X}     = _make(T, X, Y, Z)
        make(::Type{T}, X,         ::Type{Y}, Z)         where {T<:$Tupl,Y}     = _make(T, X, Y, Z)
        make(::Type{T}, ::Type{X}, ::Type{Y}, Z)         where {T<:$Tupl,X,Y}   = _make(T, X, Y, Z)
        make(::Type{T}, X,         Y,         ::Type{Z}) where {T<:$Tupl,Z}     = _make(T, X, Y, Z)
        make(::Type{T}, ::Type{X}, Y,         ::Type{Z}) where {T<:$Tupl,X,Z}   = _make(T, X, Y, Z)
        make(::Type{T}, X,         ::Type{Y}, ::Type{Z}) where {T<:$Tupl,Y,Z}   = _make(T, X, Y, Z)
        make(::Type{T}, ::Type{X}, ::Type{Y}, ::Type{Z}) where {T<:$Tupl,X,Y,Z} = _make(T, X, Y, Z)
    end
end

#### Sampler for general tuples

@generated function Sampler(RNG::Type{<:AbstractRNG}, c::Make1{T,X}, n::Repetition) where {T<:Tuple,X<:Tuple}
    @assert fieldcount(T) == fieldcount(X)
    sps = [:(sampler(RNG, c.x[$i], n)) for i in 1:length(T.parameters)]
    :(SamplerTag{Cont{T}}(tuple($(sps...))))
end

@generated function Sampler(RNG::Type{<:AbstractRNG}, ::Make0{T}, n::Repetition) where {T<:Tuple}
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

#### for "NTuple-like"

# should catch Tuple{Integer,Integer} which is not NTuple, or even Tuple{Int,UInt}, when only one sampler was passed

Sampler(RNG::Type{<:AbstractRNG}, c::Make1{T,X}, n::Repetition) where {T<:Tuple,X} =
    SamplerTag{Cont{T}}(sampler(RNG, c.x, n))

@generated function rand(rng::AbstractRNG, sp::SamplerTag{Cont{T},S}) where {T<:Tuple,S<:Sampler}
    rands = [:(convert($(T.parameters[i]), rand(rng, sp.data))) for i in 1:fieldcount(T)]
    :(tuple($(rands...)))
end

### named tuples

make(T::Type{<:NamedTuple}, args...) = _make(T, args...)

_make(::Type{NamedTuple{}}) = Make0{NamedTuple{}}()

@generated function _make(::Type{NamedTuple{K}}, X...) where {K}
    if length(X) <= 1
        NT = NamedTuple{K,_find_type(NTuple{length(K)}, X...)}
        :(Make1{$NT}(make(NTuple{length(K)}, X...)))
    else
        NT = NamedTuple{K,_find_type(Tuple, X...)}
        :(Make1{$NT}(make(Tuple, X...)))
    end
end

function _make(::Type{NamedTuple{K,V}}, X...) where {K,V}
    Make1{NamedTuple{K,V}}(make(V, X...))
end

# necessary to avoid circular defintions
Sampler(RNG::Type{<:AbstractRNG}, m::Make0{NamedTuple}, n::Repetition) =
    SamplerType{NamedTuple}()

Sampler(RNG::Type{<:AbstractRNG}, m::Make1{T}, n::Repetition) where T <: NamedTuple =
    SamplerTag{Cont{T}}(Sampler(RNG, m.x , n))

rand(rng::AbstractRNG, sp::SamplerType{NamedTuple{}}) = NamedTuple()

rand(rng::AbstractRNG, sp::SamplerTag{Cont{T}}) where T <: NamedTuple =
    T(rand(rng, sp.data))


## collections

### sets/dicts

const SetDict = Union{AbstractSet,AbstractDict}

make(T::Type{<:SetDict}, X,         n::Integer)           = Make2{find_type(T, X, n)}(X , Int(n))
make(T::Type{<:SetDict}, ::Type{X}, n::Integer) where {X} = Make2{find_type(T, X, n)}(X , Int(n))
make(T::Type{<:SetDict},            n::Integer)           = make(T, default_sampling(T), Int(n))

Sampler(RNG::Type{<:AbstractRNG}, c::Make2{T}, n::Repetition) where {T<:SetDict} =
    SamplerTag{Cont{T}}((sp = sampler(RNG, c.x, n),
                         len = c.y))

function rand(rng::AbstractRNG, sp::SamplerTag{Cont{S}}) where {S<:SetDict}
    # assuming S() creates an empty set/dict
    s = sizehint!(S(), sp.data[2])
    _rand!(rng, s, sp.data.len, sp.data.sp)
end

### sets

default_sampling(::Type{<:AbstractSet}) = Uniform(Float64)
default_sampling(::Type{<:AbstractSet{T}}) where {T} = Uniform(T)

#### Set

find_type(::Type{Set},    X, _)           = Set{val_gentype(X)}
find_type(::Type{Set{T}}, _, _) where {T} = Set{T}

### BitSet

default_sampling(::Type{BitSet}) = Uniform(Int8) # almost arbitrary, may change

find_type(::Type{BitSet}, _, _) = BitSet


### dicts

find_type(D::Type{<:AbstractDict{K,V}}, _,      ::Integer) where {K,V} = D
find_type(D::Type{<:AbstractDict{K,V}}, ::Type, ::Integer) where {K,V} = D

#### Dict/ImmutableDict

for D in (Dict, Base.ImmutableDict)
    @eval begin
        # again same inference bug
        # TODO: extend to AbstractDict ? (needs to work-around the inderence bug)
        default_sampling(::Type{$D{K,V}}) where {K,V} = Uniform(Pair{K,V})
        default_sampling(D::Type{<:$D})               = throw(ArgumentError("under-specified scalar type for $D"))

        find_type(::Type{$D{K}},           X, ::Integer) where {K} = $D{K,fieldtype(val_gentype(X), 2)}
        find_type(::Type{$D{K,V} where K}, X, ::Integer) where {V} = $D{fieldtype(val_gentype(X), 1),V}
        find_type(::Type{$D},              X, ::Integer)           = $D{fieldtype(val_gentype(X), 1),fieldtype(val_gentype(X), 2)}
    end
end

rand(rng::AbstractRNG, sp::SamplerTag{Cont{S}}) where {S<:Base.ImmutableDict} =
    foldl((d, _) -> Base.ImmutableDict(d, rand(rng, sp.data.sp)),
          1:sp.data.len,
          init=S())


### AbstractArray

default_sampling(::Type{<:AbstractArray{T}}) where {T} = Uniform(T)
default_sampling(::Type{<:AbstractArray})              = Uniform(Float64)

make(A::Type{<:AbstractArray}, X,         d1::Integer, dims::Integer...)           = make(A, X, Dims((d1, dims...)))
make(A::Type{<:AbstractArray}, ::Type{X}, d1::Integer, dims::Integer...) where {X} = make(A, X, Dims((d1, dims...)))

make(A::Type{<:AbstractArray}, dims::Dims)                    = make(A, default_sampling(A), dims)
make(A::Type{<:AbstractArray}, d1::Integer, dims::Integer...) = make(A, default_sampling(A), Dims((d1, dims...)))

if VERSION < v"1.1.0"
     # to resolve ambiguity
    make(A::Type{<:AbstractArray}, X, d1::Integer)              = make(A, X, Dims((d1,)))
    make(A::Type{<:AbstractArray}, X, d1::Integer, d2::Integer) = make(A, X, Dims((d1, d2)))
end

Sampler(RNG::Type{<:AbstractRNG}, c::Make2{A}, n::Repetition) where {A<:AbstractArray} =
    SamplerTag{A}((sampler(RNG, c.x, n), c.y))

rand(rng::AbstractRNG, sp::SamplerTag{A}) where {A<:AbstractArray} =
    rand!(rng, A(undef, sp.data[2]), sp.data[1])


#### Array

# cf. inference bug https://github.com/JuliaLang/julia/issues/28762
# we have to write out all combinations for getting proper inference
find_type(A::Type{Array{T}},           _, ::Dims{N}) where {T, N} = Array{T, N}
find_type(A::Type{Array{T,N}},         _, ::Dims{N}) where {T, N} = Array{T, N}
find_type(A::Type{Array{T,N} where T}, X, ::Dims{N}) where {N}    = Array{val_gentype(X), N}
find_type(A::Type{Array},              X, ::Dims{N}) where {N}    = Array{val_gentype(X), N}

# special shortcut

make(X,         dims::Dims)                              = make(Array, X,                       dims)
make(X,         d1::Integer, dims::Integer...)           = make(Array, X,                       Dims((d1, dims...)))
make(::Type{X}, dims::Dims)           where {X}          = make(Array, X,                       dims)
make(::Type{X}, d1::Integer, dims::Integer...) where {X} = make(Array, X,                       Dims((d1, dims...)))
make(           dims::Integer...)                        = make(Array, default_sampling(Array), Dims(dims))

# omitted: make(dims::Dims)
# for the same reason that rand(dims::Dims) doesn't produce an array, i.e. it produces a scalar picked from the tuple

#### BitArray

default_sampling(::Type{<:BitArray}) = Uniform(Bool)

find_type(::Type{BitArray{N}}, _, ::Dims{N}) where {N} = BitArray{N}
find_type(::Type{BitArray},    _, ::Dims{N}) where {N} = BitArray{N}


#### sparse vectors & matrices

find_type(::Type{SparseVector},    X, p::AbstractFloat, dims::Dims{1}) = SparseVector{   val_gentype(X), Int}
find_type(::Type{SparseMatrixCSC}, X, p::AbstractFloat, dims::Dims{2}) = SparseMatrixCSC{val_gentype(X), Int}

find_type(::Type{SparseVector{X}},    _, p::AbstractFloat, dims::Dims{1}) where {X} = SparseVector{   X, Int}
find_type(::Type{SparseMatrixCSC{X}}, _, p::AbstractFloat, dims::Dims{2}) where {X} = SparseMatrixCSC{X, Int}

# need to be explicit and split these defs in 2 (or 4) to avoid ambiguities
make(T::Type{SparseVector},    X,         p::AbstractFloat, d1::Integer)                        = make(T, X, p, Dims((d1,)))
make(T::Type{SparseVector},    ::Type{X}, p::AbstractFloat, d1::Integer)              where {X} = make(T, X, p, Dims((d1,)))
make(T::Type{SparseMatrixCSC}, X,         p::AbstractFloat, d1::Integer, d2::Integer)           = make(T, X, p, Dims((d1, d2)))
make(T::Type{SparseMatrixCSC}, ::Type{X}, p::AbstractFloat, d1::Integer, d2::Integer) where {X} = make(T, X, p, Dims((d1, d2)))

make(T::Type{SparseVector},    p::AbstractFloat, d1::Integer)              = make(T, default_sampling(T), p, Dims((d1,)))
make(T::Type{SparseMatrixCSC}, p::AbstractFloat, d1::Integer, d2::Integer) = make(T, default_sampling(T), p, Dims((d1, d2)))

make(T::Type{SparseVector},    p::AbstractFloat, dims::Dims{1}) = make(T, default_sampling(T), p, dims)
make(T::Type{SparseMatrixCSC}, p::AbstractFloat, dims::Dims{2}) = make(T, default_sampling(T), p, dims)

make(X,         p::AbstractFloat, dims::Dims{1})           = make(SparseVector, X, p, dims)
make(::Type{X}, p::AbstractFloat, dims::Dims{1}) where {X} = make(SparseVector, X, p, dims)
make(X,         p::AbstractFloat, dims::Dims{2})           = make(SparseMatrixCSC, X, p, dims)
make(::Type{X}, p::AbstractFloat, dims::Dims{2}) where {X} = make(SparseMatrixCSC, X, p, dims)

make(X,         p::AbstractFloat, d1::Integer)                        = make(X,                               p, Dims(d1))
make(X,         p::AbstractFloat, d1::Integer, d2::Integer)           = make(X,                               p, Dims((d1, d2)))
make(::Type{X}, p::AbstractFloat, d1::Integer) where {X}              = make(X,                               p, Dims(d1))
make(::Type{X}, p::AbstractFloat, d1::Integer, d2::Integer) where {X} = make(X,                               p, Dims((d1, d2)))
make(           p::AbstractFloat, dims::Dims)                         = make(default_sampling(AbstractArray), p, dims)
make(           p::AbstractFloat, d1::Integer)                        = make(default_sampling(AbstractArray), p, Dims(d1))
make(           p::AbstractFloat, d1::Integer, d2::Integer)           = make(default_sampling(AbstractArray), p, Dims((d1, d2)))

# disambiguate (away from make(String, chars, n::Integer))
make(::Type{String}, p::AbstractFloat, d1::Integer) = make(String, p, Dims(d1))

Sampler(RNG::Type{<:AbstractRNG}, c::Make3{A}, n::Repetition) where {A<:AbstractSparseArray} =
    SamplerTag{Cont{A}}((sp = sampler(RNG, c.x, n),
                         p = c.y,
                         dims = c.z))

rand(rng::AbstractRNG, sp::SamplerTag{Cont{A}}) where {A<:SparseVector} =
    sprand(rng, sp.data.dims[1], sp.data.p, (r, n)->rand(r, sp.data.sp, n))

rand(rng::AbstractRNG, sp::SamplerTag{Cont{A}}) where {A<:SparseMatrixCSC} =
    sprand(rng, sp.data.dims[1], sp.data.dims[2], sp.data.p, (r, n)->rand(r, sp.data.sp, n), gentype(sp.data.sp))


#### StaticArrays

function random_staticarrays()
    @eval using StaticArrays: tuple_length, tuple_prod, SArray, MArray
    for Arr = (:SArray, :MArray)
        @eval begin
            find_type(::Type{<:$Arr{S}}  , X) where {S<:Tuple}   = $Arr{S,val_gentype(X),tuple_length(S),tuple_prod(S)}
            find_type(::Type{<:$Arr{S,T}}, _) where {S<:Tuple,T} = $Arr{S,T,tuple_length(S),tuple_prod(S)}

            Sampler(RNG::Type{<:AbstractRNG}, c::Make1{A}, n::Repetition) where {A<:$Arr} =
                SamplerTag{Cont{A}}(Sampler(RNG, c.x, n))

            rand(rng::AbstractRNG, sp::SamplerTag{Cont{$Arr{S,T,N,L}}}) where {S,T,N,L} =
                $Arr{S,T,N,L}(rand(rng, make(NTuple{L}, sp.data)))

            @make_container(T::Type{<:$Arr})
        end
    end
end


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
