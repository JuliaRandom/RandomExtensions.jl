# definition of some distribution types


## Distribution & Combine

abstract type Distribution{T} end

Base.eltype(::Type{<:Distribution{T}}) where {T} = T

abstract type Combine{T} <: Distribution{T} end

struct Combine0{T} <: Combine{T} end

Combine(::Type{T}) where {T} = Combine0{T}()

struct Combine1{T,X} <: Combine{T}
    x::X
end

Combine(::Type{T}, x::X) where {T,X} = Combine1{T,X}(x)
Combine(::Type{T}, ::Type{X}) where {T,X} = Combine1{T,Type{X}}(X)

struct Combine2{T,X,Y} <: Combine{T}
    x::X
    y::Y
end

Combine(::Type{T}, x::X, y::Y) where {T,X,Y} = Combine2{deduce_type(T,X,Y),X,Y}(x, y)
Combine(::Type{T}, ::Type{X}, y::Y) where {T,X,Y} = Combine2{deduce_type(T,X,Y),Type{X},Y}(X, y)
Combine(::Type{T}, x::X, ::Type{Y}) where {T,X,Y} = Combine2{deduce_type(T,X,Y),X,Type{Y}}(x, Y)
Combine(::Type{T}, ::Type{X}, ::Type{Y}) where {T,X,Y} = Combine2{deduce_type(T,X,Y),Type{X},Type{Y}}(X, Y)

deduce_type(::Type{T}, ::Type{X}, ::Type{Y}) where {T,X,Y} = _deduce_type(T, Val(isconcretetype(T)), eltype(X), eltype(Y))
deduce_type(::Type{T}, ::Type{X}) where {T,X} = _deduce_type(T, Val(isconcretetype(T)), eltype(X))

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

Uniform(x::T) where {T} = UniformWrap{T,eltype(T)}(x)

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


## a dummy container type to take advangage of SamplerTag constructor

struct Cont{T} end

Base.eltype(::Type{Cont{T}}) where {T} = T
