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

struct Make3{T,X,Y,Z} <: Make{T}
    x::X
    y::Y
    z::Z
end

Make{T}(x::X,      y::Y,      z::Z)      where {T,X,Y,Z} = Make3{T,X,      Y,      Z      }(x, y, z)
Make{T}(::Type{X}, y::Y,      z::Z)      where {T,X,Y,Z} = Make3{T,Type{X},Y,      Z      }(X, y, z)
Make{T}(x::X,      ::Type{Y}, z::Z)      where {T,X,Y,Z} = Make3{T,X,      Type{Y},Z      }(x, Y, z)
Make{T}(::Type{X}, ::Type{Y}, z::Z)      where {T,X,Y,Z} = Make3{T,Type{X},Type{Y},Z      }(X, Y, z)
Make{T}(x::X,      y::Y,      ::Type{Z}) where {T,X,Y,Z} = Make3{T,X,      Y,      Type{Z}}(x, y, Z)
Make{T}(::Type{X}, y::Y,      ::Type{Z}) where {T,X,Y,Z} = Make3{T,Type{X},Y,      Type{Z}}(X, y, Z)
Make{T}(x::X,      ::Type{Y}, ::Type{Z}) where {T,X,Y,Z} = Make3{T,X,      Type{Y},Type{Z}}(x, Y, Z)
Make{T}(::Type{X}, ::Type{Y}, ::Type{Z}) where {T,X,Y,Z} = Make3{T,Type{X},Type{Y},Type{Z}}(X, Y, Z)

# for expliciteness (allows using Make3 instead of Make)
Make3{T}(x::X,      y::Y,      z::Z)      where {T,X,Y,Z} = Make3{T,X,      Y,      Z      }(x, y, z)
Make3{T}(::Type{X}, y::Y,      z::Z)      where {T,X,Y,Z} = Make3{T,Type{X},Y,      Z      }(X, y, z)
Make3{T}(x::X,      ::Type{Y}, z::Z)      where {T,X,Y,Z} = Make3{T,X,      Type{Y},Z      }(x, Y, z)
Make3{T}(::Type{X}, ::Type{Y}, z::Z)      where {T,X,Y,Z} = Make3{T,Type{X},Type{Y},Z      }(X, Y, z)
Make3{T}(x::X,      y::Y,      ::Type{Z}) where {T,X,Y,Z} = Make3{T,X,      Y,      Type{Z}}(x, y, Z)
Make3{T}(::Type{X}, y::Y,      ::Type{Z}) where {T,X,Y,Z} = Make3{T,Type{X},Y,      Type{Z}}(X, y, Z)
Make3{T}(x::X,      ::Type{Y}, ::Type{Z}) where {T,X,Y,Z} = Make3{T,X,      Type{Y},Type{Z}}(x, Y, Z)
Make3{T}(::Type{X}, ::Type{Y}, ::Type{Z}) where {T,X,Y,Z} = Make3{T,Type{X},Type{Y},Type{Z}}(X, Y, Z)


make(::Type{T}, x::X,      y::Y,      z::Z)      where {T,X,Y,Z} = Make3{find_type(T, x, y, z)}(x, y, z)
make(::Type{T}, ::Type{X}, y::Y,      z::Z)      where {T,X,Y,Z} = Make3{find_type(T, X, y, z)}(X, y, z)
make(::Type{T}, x::X,      ::Type{Y}, z::Z)      where {T,X,Y,Z} = Make3{find_type(T, x, Y, z)}(x, Y, z)
make(::Type{T}, ::Type{X}, ::Type{Y}, z::Z)      where {T,X,Y,Z} = Make3{find_type(T, X, Y, z)}(X, Y, z)
make(::Type{T}, x::X,      y::Y,      ::Type{Z}) where {T,X,Y,Z} = Make3{find_type(T, x, y, Z)}(x, y, Z)
make(::Type{T}, ::Type{X}, y::Y,      ::Type{Z}) where {T,X,Y,Z} = Make3{find_type(T, X, y, Z)}(X, y, Z)
make(::Type{T}, x::X,      ::Type{Y}, ::Type{Z}) where {T,X,Y,Z} = Make3{find_type(T, x, Y, Z)}(x, Y, Z)
make(::Type{T}, ::Type{X}, ::Type{Y}, ::Type{Z}) where {T,X,Y,Z} = Make3{find_type(T, X, Y, Z)}(X, Y, Z)


# deduce_type

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
