# definition of some distribution types


## Distribution & Make

abstract type Distribution{T} end

Base.eltype(::Type{<:Distribution{T}}) where {T} = T

struct Make{T,X<:Tuple, XX<:Tuple} <: Distribution{T}
    # XX is X with Type{A} replaced by Nothing (otherwise, X (e.g. Tuple{Type{Int}}) can't have instances)
    x::XX
end

# @inline necessary for one @inferred test on arrays
@inline Base.getindex(m::Make{T,X}, i::Int) where {T,X} =
    fieldtype(X, i) <: Type ?
        fieldtype(X, i).parameters[1] :
        m.x[i]

@generated function Make{T}(X...) where T
    XX = Tuple{(x <: Type ? Nothing : x for x in X)...}
    y = [x <: Type ? nothing : :(X[$i]) for (i, x) in enumerate(X)]
    :(Make{T,Tuple{$X...},$XX}(tuple($(y...))))
end

Make0{T} = Make{T,Tuple{}}
Make1{T} = Make{T,Tuple{X}}     where X
Make2{T} = Make{T,Tuple{X,Y}}   where {X, Y}
Make3{T} = Make{T,Tuple{X,Y,Z}} where {X, Y, Z}

Make0{T}()        where {T} = Make{T}()
Make1{T}(x)       where {T} = Make{T}(x)
Make2{T}(x, y)    where {T} = Make{T}(x, y)
Make3{T}(x, y, z) where {T} = Make{T}(x, y, z)

# default maketype & make
maketype(::Type{T}, x...) where {T} = T

make(::Type{T}, x...) where {T} = Make{maketype(T, x...)}(x...)

find_deduced_type(::Type{T}, ::X,     ) where {T,X} = deduce_type(T, gentype(X))
find_deduced_type(::Type{T}, ::Type{X}) where {T,X} = deduce_type(T, X)

find_deduced_type(::Type{T}, ::X,       ::Y)       where {T,X,Y} = deduce_type(T, gentype(X), gentype(Y))
find_deduced_type(::Type{T}, ::Type{X}, ::Y)       where {T,X,Y} = deduce_type(T, X,          gentype(Y))
find_deduced_type(::Type{T}, ::X,       ::Type{Y}) where {T,X,Y} = deduce_type(T, gentype(X), Y)
find_deduced_type(::Type{T}, ::Type{X}, ::Type{Y}) where {T,X,Y} = deduce_type(T, X,          Y)

deduce_type(::Type{T}, ::Type{X}, ::Type{Y}) where {T,X,Y} = _deduce_type(T, Val(isconcretetype(T)), X, Y)
deduce_type(::Type{T}, ::Type{X}) where {T,X} = _deduce_type(T, Val(isconcretetype(T)), X)

_deduce_type(::Type{T}, ::Val{true},  ::Type{X}, ::Type{Y}) where {T,X,Y} = T
_deduce_type(::Type{T}, ::Val{false}, ::Type{X}, ::Type{Y}) where {T,X,Y} = deduce_type(T{X}, Y)

_deduce_type(::Type{T}, ::Val{true},  ::Type{X}) where {T,X} = T
_deduce_type(::Type{T}, ::Val{false}, ::Type{X}) where {T,X} = T{X}


## Const

# distribution always yielding the same value
struct Const{T} <: Distribution{T}
    x::T
end

Base.getindex(c::Const) = c.x

rand(::AbstractRNG, c::SamplerTrivial{<:Const}) = c[][]


## Uniform

abstract type Uniform{T} <: Distribution{T} end


struct UniformType{T} <: Uniform{T} end

Uniform(::Type{T}) where {T} = UniformType{T}()

Base.getindex(::UniformType{T}) where {T} = T

struct UniformWrap{T,E} <: Uniform{E}
    val::T
end

Uniform(x::T) where {T} = UniformWrap{T,gentype(T)}(x)

Base.getindex(x::UniformWrap) = x.val


## Normal & Exponential

abstract type Normal{T} <: Distribution{T} end

struct Normal01{T} <: Normal{T} end

struct Normalμσ{T} <: Normal{T}
    μ::T
    σ::T
end

const NormalTypes = Union{AbstractFloat,Complex{<:AbstractFloat}}

Normal(::Type{T}=Float64) where {T<:NormalTypes} = Normal01{T}()
Normal(μ::T, σ::T) where {T<:NormalTypes} = Normalμσ(μ, σ)
Normal(μ::T, σ::T) where {T<:Real} = Normalμσ(AbstractFloat(μ), AbstractFloat(σ))
Normal(μ, σ) = Normal(promote(μ, σ)...)

abstract type Exponential{T} <: Distribution{T} end

struct Exponential1{T} <: Exponential{T} end

struct Exponentialθ{T} <: Exponential{T}
    θ::T
end

Exponential(::Type{T}=Float64) where {T<:AbstractFloat} = Exponential1{T}()
Exponential(θ::T) where {T<:AbstractFloat} = Exponentialθ(θ)
Exponential(θ::Real) = Exponentialθ(AbstractFloat(θ))


## floats

abstract type FloatInterval{T<:AbstractFloat} <: Uniform{T} end

abstract type CloseOpen{ T<:AbstractFloat} <: FloatInterval{T} end
abstract type OpenClose{ T<:AbstractFloat} <: FloatInterval{T} end
abstract type CloseClose{T<:AbstractFloat} <: FloatInterval{T} end
abstract type OpenOpen{  T<:AbstractFloat} <: FloatInterval{T} end

struct CloseOpen12{T<:AbstractFloat} <: CloseOpen{T} end # interval [1,2)

struct CloseOpen01{ T<:AbstractFloat} <: CloseOpen{T}  end # interval [0,1)
struct OpenClose01{ T<:AbstractFloat} <: OpenClose{T}  end # interval (0,1]
struct CloseClose01{T<:AbstractFloat} <: CloseClose{T} end # interval [0,1]
struct OpenOpen01{  T<:AbstractFloat} <: OpenOpen{T}   end # interval (0,1)

struct CloseOpenAB{T<:AbstractFloat} <: CloseOpen{T} # interval [a,b)
    a::T
    b::T

    CloseOpenAB{T}(a::T, b::T) where {T} = (check_interval(a, b); new{T}(a, b))
end

struct OpenCloseAB{T<:AbstractFloat} <: OpenClose{T} # interval (a,b]
    a::T
    b::T

    OpenCloseAB{T}(a::T, b::T) where {T} = (check_interval(a, b); new{T}(a, b))
end

struct CloseCloseAB{T<:AbstractFloat} <: CloseClose{T} # interval [a,b]
    a::T
    b::T

    CloseCloseAB{T}(a::T, b::T) where {T} = (check_interval(a, b); new{T}(a, b))
end

struct OpenOpenAB{T<:AbstractFloat} <: OpenOpen{T} # interval (a,b)
    a::T
    b::T

    OpenOpenAB{T}(a::T, b::T) where {T} = (check_interval(a, b); new{T}(a, b))
end

check_interval(a, b) = a >= b && throw(ArgumentError("invalid interval specification"))

const FloatInterval_64 = FloatInterval{Float64}
const CloseOpen01_64   = CloseOpen01{Float64}
const CloseOpen12_64   = CloseOpen12{Float64}

CloseOpen01(::Type{T}=Float64) where {T<:AbstractFloat} = CloseOpen01{T}()
CloseOpen12(::Type{T}=Float64) where {T<:AbstractFloat} = CloseOpen12{T}()

CloseOpen(::Type{T}=Float64) where {T<:AbstractFloat} = CloseOpen01{T}()
CloseOpen(a::T, b::T) where {T<:AbstractFloat} = CloseOpenAB{T}(a, b)

OpenClose(::Type{T}=Float64) where {T<:AbstractFloat} = OpenClose01{T}()
OpenClose(a::T, b::T) where {T<:AbstractFloat} = OpenCloseAB{T}(a, b)

CloseClose(::Type{T}=Float64) where {T<:AbstractFloat} = CloseClose01{T}()
CloseClose(a::T, b::T) where {T<:AbstractFloat} = CloseCloseAB{T}(a, b)

OpenOpen(::Type{T}=Float64) where {T<:AbstractFloat} = OpenOpen01{T}()
OpenOpen(a::T, b::T) where {T<:AbstractFloat} = OpenOpenAB{T}(a, b)

# convenience functions

CloseOpen(a, b) = CloseOpen(promote(a, b)...)
CloseOpen(a::T, b::T) where {T} = CloseOpen(AbstractFloat(a), AbstractFloat(b))

OpenClose(a, b) = OpenClose(promote(a, b)...)
OpenClose(a::T, b::T) where {T} = OpenClose(AbstractFloat(a), AbstractFloat(b))

CloseClose(a, b) = CloseClose(promote(a, b)...)
CloseClose(a::T, b::T) where {T} = CloseClose(AbstractFloat(a), AbstractFloat(b))

OpenOpen(a, b) = OpenOpen(promote(a, b)...)
OpenOpen(a::T, b::T) where {T} = OpenOpen(AbstractFloat(a), AbstractFloat(b))

## Bernoulli

struct Bernoulli{T<:Number} <: Distribution{T}
    p::Float64

    Bernoulli{T}(p::Real) where {T} = let pf = Float64(p)
        0.0 <= pf <= 1.0 ? new(pf) :
            throw(DomainError(p, "Bernoulli: parameter p must satisfy 0.0 <= p <= 1.0"))
    end
end

Bernoulli(p::Real=0.5) = Bernoulli(Int, p)
Bernoulli(::Type{T}, p::Real=0.5) where {T<:Number} = Bernoulli{T}(p)
