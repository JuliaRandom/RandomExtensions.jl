# definition of samplers and random generation


# allows to call `Sampler` only when the the arg isn't a Sampler itself
sampler(::Type{RNG}, X,          n::Repetition=Val(Inf)) where {RNG<:AbstractRNG} =
    Sampler(RNG, X, n)

sampler(::Type{RNG}, ::Type{X},  n::Repetition=Val(Inf)) where {RNG<:AbstractRNG,X} =
    Sampler(RNG, X, n)

sampler(::Type{RNG}, X::Sampler, n::Repetition=Val(Inf)) where {RNG<:AbstractRNG} = X

sampler(rng::AbstractRNG, X, n::Repetition=Val(Inf)) = sampler(typeof(rng), X, n)


## defaults

### 0-arg

make() = make(Float64)

### type: handles e.g. rand(make(Int))

# rand(rng, ::SamplerType{Make0{X}}) should not be overloaded, as make(T)
# has this special pass-thru Sampler defined below

Sampler(::Type{RNG}, ::Make0{X}, n::Repetition) where {RNG<:AbstractRNG,X} =
    Sampler(RNG, X, n)

### object: handles e.g. rand(make(1:3))

# make(x) where x isn't a type should create a distribution d such that rand(x) is
# equivalent to rand(d); so we need to overload make(x) to give a different type
# than Make1{gentype(x)}, as we can't simply define a generic pass-thru Sampler for
# that Make1 type (because users expect the default to be a SamplerTrivial{<:Make1{...}},
# and might want to define only a rand method on this SamplerTrivial)

# like Make1
struct MakeWrap{T,X} <: Distribution{T}
    x::X
end

# make(::Type) is intercepted in distribution.jl
make(x) = MakeWrap{gentype(x),typeof(x)}(x)

Sampler(::Type{RNG}, x::MakeWrap, n::Repetition) where {RNG<:AbstractRNG} =
    Sampler(RNG, x.x, n)


## Uniform

Sampler(::Type{RNG}, d::Union{UniformWrap,UniformType}, n::Repetition
        ) where {RNG<:AbstractRNG} =
            Sampler(RNG, d[], n)


## floats

### fall-back on Random definitions

for CO in (:CloseOpen01, :CloseOpen12)
    @eval begin
        Sampler(::Type{RNG}, ::$CO{T}, n::Repetition) where {RNG<:AbstractRNG,T} =
            Sampler(RNG, Random.$CO{T}(), n)

        Sampler(::Type{<:AbstractRNG}, ::$CO{BigFloat}, ::Repetition) =
            Random.SamplerBigFloat{Random.$CO{BigFloat}}(precision(BigFloat))
    end
end

### new intervals 01

# TODO: optimize for BigFloat

for CO = (:OpenClose01, :OpenOpen01, :CloseClose01)
    @eval Sampler(::Type{RNG}, I::$CO{T}, n::Repetition) where {RNG<:AbstractRNG,T} =
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

    @eval Sampler(::Type{RNG}, d::$CO{T}, n::Repetition) where {RNG<:AbstractRNG,T} =
        SamplerTag{$CO{T}}((a=d.a, d=d.b - d.a, sp=Sampler(RNG, $CO01{T}(), n)))

    @eval rand(rng::AbstractRNG, sp::SamplerTag{$CO{T}}) where {T} =
        sp.data.a + sp.data.d  * rand(rng, sp.data.sp)
end

## Normal & Exponential

rand(rng::AbstractRNG, ::SamplerTrivial{Normal01{T}}) where {T<:NormalTypes} =
    randn(rng, T)

Sampler(::Type{RNG}, d::Normalμσ{T}, n::Repetition) where {RNG<:AbstractRNG,T} =
    SamplerSimple(d, Sampler(RNG, Normal(T), n))

rand(rng::AbstractRNG, sp::SamplerSimple{Normalμσ{T},<:Sampler}) where {T} =
    sp[].μ + sp[].σ  * rand(rng, sp.data)

rand(rng::AbstractRNG, ::SamplerTrivial{Exponential1{T}}) where {T<:AbstractFloat} =
    randexp(rng, T)

Sampler(::Type{RNG}, d::Exponentialθ{T}, n::Repetition) where {RNG<:AbstractRNG,T} =
    SamplerSimple(d, Sampler(RNG, Exponential(T), n))

rand(rng::AbstractRNG, sp::SamplerSimple{Exponentialθ{T},<:Sampler}) where {T} =
    sp[].θ * rand(rng, sp.data)


## Bernoulli

Sampler(::Type{RNG}, b::Bernoulli, n::Repetition) where {RNG<:AbstractRNG} =
    SamplerTag{typeof(b)}(b.p+1.0)

rand(rng::AbstractRNG, sp::SamplerTag{Bernoulli{T}}) where {T} =
    ifelse(rand(rng, CloseOpen12()) < sp.data, one(T), zero(T))


## Categorical

Sampler(::Type{RNG}, c::Categorical, n::Repetition) where {RNG<:AbstractRNG} =
    SamplerSimple(c, Sampler(RNG, CloseOpen(), n))

# unfortunately requires @inline to avoid allocating
@inline rand(rng::AbstractRNG, sp::SamplerSimple{Categorical{T}}) where {T} =
    let c = rand(rng, sp.data)
        T(findfirst(x -> x >= c, sp[].cdf))
    end

# NOTE:
# if length(cdf) is somewhere between 150 and 200, the following gets faster:
#   T(searchsortedfirst(sp[].cdf, rand(rng, sp.data)))


## random elements from pairs

#= disabled in favor of a special meaning for pairs

Sampler(RNG::Type{<:AbstractRNG}, t::Pair, n::Repetition) =
    SamplerSimple(t, Sampler(RNG, Bool, n))

rand(rng::AbstractRNG, sp::SamplerSimple{<:Pair}) =
    @inbounds return sp[][1 + rand(rng, sp.data)]
=#

## composite types

### sampler for pairs and complex numbers

maketype(::Type{Pair},              x, y)             = Pair{val_gentype(x), val_gentype(y)}
maketype(::Type{Pair{X}},           _, y) where {X}   = Pair{X, val_gentype(y)}
maketype(::Type{Pair{X,Y} where X}, x, _) where {Y}   = Pair{val_gentype(x), Y}
maketype(::Type{Pair{X,Y}},         _, _) where {X,Y} = Pair{X,Y}

maketype(::Type{Complex},    x) = Complex{val_gentype(x)}
maketype(::Type{T}, _) where {T<:Complex} = T

maketype(::Type{Complex},    x, y) = Complex{promote_type(val_gentype(x), val_gentype(y))}
maketype(::Type{T}, _, _) where {T<:Complex} = T

function Sampler(::Type{RNG}, u::Make2{T}, n::Repetition
                 ) where RNG<:AbstractRNG where T <: Union{Pair,Complex}
    sp1 = sampler(RNG, u[1], n)
    sp2 = u[1] == u[2] ? sp1 : sampler(RNG, u[2], n)
    SamplerTag{Cont{T}}((sp1, sp2))
end

rand(rng::AbstractRNG, sp::SamplerTag{Cont{T}}) where {T<:Union{Pair,Complex}} =
    T(rand(rng, sp.data[1]), rand(rng, sp.data[2]))


#### additional convenience methods

# rand(Pair{A,B}) => rand(make(Pair{A,B}, A, B))
if VERSION < v"1.11.0-DEV.618" # now implemented in `Random`
    Sampler(::Type{RNG}, ::Type{Pair{A,B}}, n::Repetition) where {RNG<:AbstractRNG,A,B} =
        Sampler(RNG, make(Pair{A,B}, A, B), n)
end

# rand(make(Complex, x)) => rand(make(Complex, x, x))
Sampler(::Type{RNG}, u::Make1{T}, n::Repetition) where {RNG<:AbstractRNG,T<:Complex} =
    Sampler(RNG, make(T, u[1], u[1]), n)

# rand(Complex{T}) => rand(make(Complex{T}, T, T)) (redundant with implem in Random)
Sampler(::Type{RNG}, ::Type{Complex{T}}, n::Repetition) where {RNG<:AbstractRNG,T<:Real} =
    Sampler(RNG, make(Complex{T}, T, T), n)


### sampler for tuples

#### "simple scalar" (non-make) version

Sampler(::Type{RNG}, ::Type{T}, n::Repetition
        ) where {RNG<:AbstractRNG,T<:Union{Tuple,NamedTuple}} =
            Sampler(RNG, make(T), n)

if VERSION >= v"1.11.0-DEV.573"
    # now `Random` implements `rand(Tuple{...})`, so be more specific for
    # special stuff still not implemented by `Random`
    # TODO: we should probably remove this
    Sampler(::Type{RNG}, ::Type{Tuple}, n::Repetition) where {RNG <: AbstractRNG} =
        Sampler(RNG, make(Tuple), n)

    Sampler(::Type{RNG}, ::Type{NTuple{N}}, n::Repetition) where {RNG <: AbstractRNG, N} =
        Sampler(RNG, make(NTuple{N}), n)
end

#### make

# implement make(Tuple, S1, S2...), e.g. for rand(make(Tuple, Int, 1:3)),
# and       make(NTuple{N}, S)

_maketype(::Type{T}) where {T<:Tuple} =
    T === Tuple ?
        Tuple{} :
    T === NTuple ?
        Tuple{} :
    T isa UnionAll && Type{T} <: Type{NTuple{N}} where N ?
        T{default_gentype(Tuple)} :
        T

function _maketype(::Type{T}, args...) where T <: Tuple
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
    isempty(args) && return :(Make0{$(_maketype(T))}())
    TT = _maketype(T, args...)
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

make(::Type{T}, args...) where {T<:Tuple} = _make(T, args...)

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

@generated function Sampler(::Type{RNG}, c::Make1{T,X}, n::Repetition
                            ) where {RNG<:AbstractRNG,T<:Tuple,X<:Tuple}
    @assert fieldcount(T) == fieldcount(X)
    sps = [:(sampler(RNG, c[1][$i], n)) for i in 1:length(T.parameters)]
    :(SamplerTag{Cont{T}}(tuple($(sps...))))
end

@generated function Sampler(::Type{RNG}, ::Make0{T}, n::Repetition
                            ) where {RNG<:AbstractRNG,T<:Tuple}
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

Sampler(::Type{RNG}, c::Make1{T,X}, n::Repetition) where {RNG<:AbstractRNG,T<:Tuple,X} =
    SamplerTag{Cont{T}}(sampler(RNG, c[1], n))

@generated function rand(rng::AbstractRNG, sp::SamplerTag{Cont{T},S}) where {T<:Tuple,S<:Sampler}
    rands = [:(convert($(T.parameters[i]), rand(rng, sp.data))) for i in 1:fieldcount(T)]
    :(tuple($(rands...)))
end

### named tuples

make(::Type{T}, args...) where {T<:NamedTuple} = _make(T, args...)

_make(::Type{NamedTuple{}}) = Make0{NamedTuple{}}()

@generated function _make(::Type{NamedTuple{K}}, X...) where {K}
    if length(X) <= 1
        NT = NamedTuple{K,_maketype(NTuple{length(K)}, X...)}
        :(Make1{$NT}(make(NTuple{length(K)}, X...)))
    else
        NT = NamedTuple{K,_maketype(Tuple, X...)}
        :(Make1{$NT}(make(Tuple, X...)))
    end
end

function _make(::Type{NamedTuple{K,V}}, X...) where {K,V}
    Make1{NamedTuple{K,V}}(make(V, X...))
end

# necessary to avoid circular defintions
Sampler(::Type{RNG}, m::Make0{NamedTuple}, n::Repetition) where {RNG<:AbstractRNG} =
    SamplerType{NamedTuple}()

Sampler(::Type{RNG}, m::Make1{T}, n::Repetition) where {RNG<:AbstractRNG, T <: NamedTuple} =
    SamplerTag{Cont{T}}(Sampler(RNG, m[1] , n))

rand(rng::AbstractRNG, sp::SamplerType{NamedTuple{}}) = NamedTuple()

rand(rng::AbstractRNG, sp::SamplerTag{Cont{T}}) where T <: NamedTuple =
    T(rand(rng, sp.data))


## collections

### sets/dicts

const SetDict = Union{AbstractSet,AbstractDict}

make(::Type{T}, X,         n::Integer) where {T<:SetDict}   = Make2{maketype(T, X, n)}(X , Int(n))
make(::Type{T}, ::Type{X}, n::Integer) where {T<:SetDict,X} = Make2{maketype(T, X, n)}(X , Int(n))
make(::Type{T},            n::Integer) where {T<:SetDict}   = make(T, default_sampling(T), Int(n))

Sampler(::Type{RNG}, c::Make2{T}, n::Repetition) where {RNG<:AbstractRNG,T<:SetDict} =
    SamplerTag{Cont{T}}((sp = sampler(RNG, c[1], n),
                         len = c[2]))

function rand(rng::AbstractRNG, sp::SamplerTag{Cont{S}}) where {S<:SetDict}
    # assuming S() creates an empty set/dict
    s = sizehint!(S(), sp.data[2])
    _rand!(rng, s, sp.data.len, sp.data.sp)
end

### sets

default_sampling(::Type{<:AbstractSet}) = Uniform(Float64)
default_sampling(::Type{<:AbstractSet{T}}) where {T} = Uniform(T)

#### Set

maketype(::Type{Set},    X, _)           = Set{val_gentype(X)}
maketype(::Type{Set{T}}, _, _) where {T} = Set{T}

### BitSet

default_sampling(::Type{BitSet}) = Uniform(Int8) # almost arbitrary, may change

maketype(::Type{BitSet}, _, _) = BitSet


### dicts

# K,V parameters are necessary here
maketype(::Type{D}, _,      ::Integer) where {K,V,D<:AbstractDict{K,V}} = D
maketype(::Type{D}, ::Type, ::Integer) where {K,V,D<:AbstractDict{K,V}} = D

#### Dict/ImmutableDict

for D in (Dict, Base.ImmutableDict)
    @eval begin
        # again same inference bug
        # TODO: extend to AbstractDict ? (needs to work-around the inderence bug)
        default_sampling(::Type{$D{K,V}}) where {K,V} = Uniform(Pair{K,V})
        default_sampling(D::Type{<:$D})               = throw(ArgumentError("under-specified scalar type for $D"))

        maketype(::Type{$D{K}},           X, ::Integer) where {K} = $D{K,fieldtype(val_gentype(X), 2)}
        maketype(::Type{$D{K,V} where K}, X, ::Integer) where {V} = $D{fieldtype(val_gentype(X), 1),V}
        maketype(::Type{$D},              X, ::Integer)           = $D{fieldtype(val_gentype(X), 1),fieldtype(val_gentype(X), 2)}
    end
end

rand(rng::AbstractRNG, sp::SamplerTag{Cont{S}}) where {S<:Base.ImmutableDict} =
    foldl((d, _) -> Base.ImmutableDict(d, rand(rng, sp.data.sp)),
          1:sp.data.len,
          init=S())


### AbstractArray

default_sampling(::Type{<:AbstractArray{T}}) where {T} = Uniform(T)
default_sampling(::Type{<:AbstractArray})              = Uniform(Float64)

make(::Type{A}, X,         d1::Integer, dims::Integer...) where {A<:AbstractArray} =
    make(A, X, Dims((d1, dims...)))

make(::Type{A}, ::Type{X}, d1::Integer, dims::Integer...) where {A<:AbstractArray,X} =
    make(A, X, Dims((d1, dims...)))

make(::Type{A}, dims::Dims)                    where {A<:AbstractArray} =
    make(A, default_sampling(A), dims)

make(::Type{A}, d1::Integer, dims::Integer...) where {A<:AbstractArray} =
    make(A, default_sampling(A), Dims((d1, dims...)))

if VERSION < v"1.1.0"
     # to resolve ambiguity
    make(A::Type{<:AbstractArray}, X, d1::Integer)              = make(A, X, Dims((d1,)))
    make(A::Type{<:AbstractArray}, X, d1::Integer, d2::Integer) = make(A, X, Dims((d1, d2)))
end

Sampler(::Type{RNG}, c::Make2{A}, n::Repetition) where {RNG<:AbstractRNG,A<:AbstractArray} =
    SamplerTag{A}((sampler(RNG, c[1], n), c[2]))

rand(rng::AbstractRNG, sp::SamplerTag{A}) where {A<:AbstractArray} =
    rand!(rng, A(undef, sp.data[2]), sp.data[1])


#### Array

# cf. inference bug https://github.com/JuliaLang/julia/issues/28762
# we have to write out all combinations for getting proper inference
maketype(::Type{Array{T}},           _, ::Dims{N}) where {T, N} = Array{T, N}
maketype(::Type{Array{T,N}},         _, ::Dims{N}) where {T, N} = Array{T, N}
maketype(::Type{Array{T,N} where T}, X, ::Dims{N}) where {N}    = Array{val_gentype(X), N}
maketype(::Type{Array},              X, ::Dims{N}) where {N}    = Array{val_gentype(X), N}

#### BitArray

default_sampling(::Type{<:BitArray}) = Uniform(Bool)

maketype(::Type{BitArray{N}}, _, ::Dims{N}) where {N} = BitArray{N}
maketype(::Type{BitArray},    _, ::Dims{N}) where {N} = BitArray{N}


#### sparse vectors & matrices

maketype(::Type{SparseVector},    X, p::AbstractFloat, dims::Dims{1}) = SparseVector{   val_gentype(X), Int}
maketype(::Type{SparseMatrixCSC}, X, p::AbstractFloat, dims::Dims{2}) = SparseMatrixCSC{val_gentype(X), Int}

maketype(::Type{SparseVector{X}},    _, p::AbstractFloat, dims::Dims{1}) where {X} = SparseVector{   X, Int}
maketype(::Type{SparseMatrixCSC{X}}, _, p::AbstractFloat, dims::Dims{2}) where {X} = SparseMatrixCSC{X, Int}

# need to be explicit and split these defs in 2 (or 4) to avoid ambiguities
# TODO: check that using T instead of SparseVector in the RHS doesn't have perfs issues
make(T::Type{SparseVector},    X,         p::AbstractFloat, d1::Integer)                        = make(T, X, p, Dims((d1,)))
make(T::Type{SparseVector},    ::Type{X}, p::AbstractFloat, d1::Integer)              where {X} = make(T, X, p, Dims((d1,)))
make(T::Type{SparseMatrixCSC}, X,         p::AbstractFloat, d1::Integer, d2::Integer)           = make(T, X, p, Dims((d1, d2)))
make(T::Type{SparseMatrixCSC}, ::Type{X}, p::AbstractFloat, d1::Integer, d2::Integer) where {X} = make(T, X, p, Dims((d1, d2)))

make(T::Type{SparseVector},    p::AbstractFloat, d1::Integer)              = make(T, default_sampling(T), p, Dims((d1,)))
make(T::Type{SparseMatrixCSC}, p::AbstractFloat, d1::Integer, d2::Integer) = make(T, default_sampling(T), p, Dims((d1, d2)))

make(T::Type{SparseVector},    p::AbstractFloat, dims::Dims{1}) = make(T, default_sampling(T), p, dims)
make(T::Type{SparseMatrixCSC}, p::AbstractFloat, dims::Dims{2}) = make(T, default_sampling(T), p, dims)


Sampler(::Type{RNG}, c::Make3{A}, n::Repetition) where {RNG<:AbstractRNG,A<:AbstractSparseArray} =
    SamplerTag{Cont{A}}((sp = sampler(RNG, c[1], n),
                         p = c[2],
                         dims = c[3]))

rand(rng::AbstractRNG, sp::SamplerTag{Cont{A}}) where {A<:SparseVector} =
    sprand(rng, sp.data.dims[1], sp.data.p, (r, n)->rand(r, sp.data.sp, n))

rand(rng::AbstractRNG, sp::SamplerTag{Cont{A}}) where {A<:SparseMatrixCSC} =
    sprand(rng, sp.data.dims[1], sp.data.dims[2], sp.data.p, (r, n)->rand(r, sp.data.sp, n), gentype(sp.data.sp))


#### StaticArrays

function random_staticarrays()
    @eval using StaticArrays: tuple_length, tuple_prod, SArray, MArray
    for Arr = (:SArray, :MArray)
        @eval begin
            maketype(::Type{<:$Arr{S}}  , X) where {S<:Tuple}   = $Arr{S,val_gentype(X),tuple_length(S),tuple_prod(S)}
            maketype(::Type{<:$Arr{S,T}}, _) where {S<:Tuple,T} = $Arr{S,T,tuple_length(S),tuple_prod(S)}

            Sampler(::Type{RNG}, c::Make1{A}, n::Repetition) where {RNG<:AbstractRNG,A<:$Arr} =
                SamplerTag{Cont{A}}(Sampler(RNG, c[1], n))

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

    Sampler(::Type{RNG}, ::Type{String}, n::Repetition) where {RNG<:AbstractRNG} =
        SamplerTag{Cont{String}}((RNG === MersenneTwister ? s : Sampler(RNG, b, n)) => 8)

    function Sampler(::Type{RNG}, c::Make2{String}, n::Repetition) where {RNG<:AbstractRNG}
        sp = RNG === MersenneTwister && c[2] === b ?
            s : sampler(RNG, c[2], n)
        SamplerTag{Cont{String}}(sp => c[1])
    end

    rand(rng::AbstractRNG, sp::SamplerTag{Cont{String}}) = String(rand(rng, sp.data.first, sp.data.second))
end


## X => a / X => (a, as...) syntax as an alternative to make(X, a) / make(X, a, as...)

# this is experimental

pair_to_make((a, b)::Pair) =
    b isa Tuple ?
        make(a, map(pair_to_make, b)...) :
        make(a, pair_to_make(b))

pair_to_make(x) = x

@inline Sampler(::Type{RNG}, p::Pair, r::Repetition) where {RNG<:AbstractRNG} =
    Sampler(RNG, pair_to_make(p), r)

# nothing can be inferred when only the pair type is available
@inline gentype(::Type{<:Pair}) = Any

@inline gentype(p::Pair) = gentype(pair_to_make(p))
