# definition of some distribution types


## Distribution & Make

abstract type Distribution{T} end

Base.eltype(::Type{<:Distribution{T}}) where {T} = T

abstract type Make{T} <: Distribution{T} end

struct Make0{T} <: Make{T} end

Make( ::Type{T}) where {T} = Make0{T}()
Make0(::Type{T}) where {T} = Make0{T}()
make(::Type{T}) where {T} = Make0{find_type(T)}()

struct Make1{T,X} <: Make{T}
    x::X
end

Make{T}(x::X)      where {T,X} = Make1{T,X}(      x)
Make{T}(::Type{X}) where {T,X} = Make1{T,Type{X}}(X)

# for expliciteness (allows using Make1 instead of Make)
Make1{T}(x::X)      where {T,X} = Make1{T,X}(      x)
Make1{T}(::Type{X}) where {T,X} = Make1{T,Type{X}}(X)

make(::Type{T}, x::X)      where {T,X} = Make{find_type(T,x)}(x)
make(::Type{T}, ::Type{X}) where {T,X} = Make{find_type(T,X)}(X)

find_deduced_type(::Type{T}, ::X,     ) where {T,X} = deduce_type(T, gentype(X))
find_deduced_type(::Type{T}, ::Type{X}) where {T,X} = deduce_type(T, X)

struct Make2{T,X,Y} <: Make{T}
    x::X
    y::Y
end

Make{T}(x::X,      y::Y)      where {T,X,Y} = Make2{T,X,      Y}(      x, y)
Make{T}(::Type{X}, y::Y)      where {T,X,Y} = Make2{T,Type{X},Y}(      X, y)
Make{T}(x::X,      ::Type{Y}) where {T,X,Y} = Make2{T,X,      Type{Y}}(x, Y)
Make{T}(::Type{X}, ::Type{Y}) where {T,X,Y} = Make2{T,Type{X},Type{Y}}(X, Y)

# for expliciteness (allows using Make2 instead of Make)
Make2{T}(x::X,      y::Y)      where {T,X,Y} = Make2{T,X,      Y}(      x, y)
Make2{T}(::Type{X}, y::Y)      where {T,X,Y} = Make2{T,Type{X},Y}(      X, y)
Make2{T}(x::X,      ::Type{Y}) where {T,X,Y} = Make2{T,X,      Type{Y}}(x, Y)
Make2{T}(::Type{X}, ::Type{Y}) where {T,X,Y} = Make2{T,Type{X},Type{Y}}(X, Y)

make(::Type{T}, x::X,      y::Y)      where {T,X,Y} = Make{find_type(T,x,y)}(x, y)
make(::Type{T}, ::Type{X}, y::Y)      where {T,X,Y} = Make{find_type(T,X,y)}(X, y)
make(::Type{T}, x::X,      ::Type{Y}) where {T,X,Y} = Make{find_type(T,x,Y)}(x, Y)
make(::Type{T}, ::Type{X}, ::Type{Y}) where {T,X,Y} = Make{find_type(T,X,Y)}(X, Y)

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

Normal(::Type{T}=Float64) where {T} = Normal01{T}()
Normal(μ::T, σ::T) where {T} = Normalμσ(μ, σ)

abstract type Exponential{T} <: Distribution{T} end

struct Exponential1{T} <: Exponential{T} end

struct Exponentialθ{T} <: Exponential{T}
    θ::T
end

Exponential(::Type{T}=Float64) where {T<:AbstractFloat} = Exponential1{T}()
Exponential(θ::T) where {T<:AbstractFloat} = Exponentialθ(θ)


## floats

abstract type FloatInterval{T<:AbstractFloat} <: Uniform{T} end
abstract type CloseOpen{T<:AbstractFloat} <: FloatInterval{T} end

struct CloseOpen01{T<:AbstractFloat} <: CloseOpen{T} end # interval [0,1)
struct CloseOpen12{T<:AbstractFloat} <: CloseOpen{T} end # interval [1,2)

struct CloseOpenAB{T<:AbstractFloat} <: CloseOpen{T} # interval [a,b)
    a::T
    b::T
end

const FloatInterval_64 = FloatInterval{Float64}
const CloseOpen01_64   = CloseOpen01{Float64}
const CloseOpen12_64   = CloseOpen12{Float64}

CloseOpen01(::Type{T}=Float64) where {T<:AbstractFloat} = CloseOpen01{T}()
CloseOpen12(::Type{T}=Float64) where {T<:AbstractFloat} = CloseOpen12{T}()

CloseOpen(::Type{T}=Float64) where {T<:AbstractFloat} = CloseOpen01{T}()
CloseOpen(a::T, b::T) where {T<:AbstractFloat} = CloseOpenAB{T}(a, b)


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
