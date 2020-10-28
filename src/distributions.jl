# definition of some distribution types


## Distribution & Make

abstract type Distribution{T} end

Base.eltype(::Type{<:Distribution{T}}) where {T} = T

struct Make{T,X<:Tuple, XX<:Tuple} <: Distribution{T}
    # XX is X with Type{A} replaced by Nothing (otherwise, X (e.g. Tuple{Type{Int}}) can't have instances)
    x::XX
end

# @inline necessary for one @inferred test on arrays
@inline function Base.getindex(m::Make{T,X}, i::Integer) where {T,X}
    i = Int(i)
    fieldtype(X, i) <: Type ?
        fieldtype(X, i).parameters[1] :
        m.x[i]
end

@inline Base.getindex(m::Make, idxs::AbstractVector{<:Integer}) =
    ntuple(i->m[idxs[i]], length(idxs))
# seems faster than `Tuple(m[i] for i in idxs)`

Base.lastindex(m::Make) = lastindex(m.x)

@generated function Make{T}(X...) where T
    XX = Tuple{(x <: Type ? Nothing : x for x in X)...}
    y = [x <: Type ? nothing : :(X[$i]) for (i, x) in enumerate(X)]
    :(Make{T,Tuple{$X...},$XX}(tuple($(y...))))
end

Make0{T} = Make{T,Tuple{}}
Make1{T} = Make{T,Tuple{X}}         where X
Make2{T} = Make{T,Tuple{X,Y}}       where {X, Y}
Make3{T} = Make{T,Tuple{X,Y,Z}}     where {X, Y, Z}
Make4{T} = Make{T,Tuple{X,Y,Z,U}}   where {X, Y, Z, U}
Make5{T} = Make{T,Tuple{X,Y,Z,U,V}} where {X, Y, Z, U, V}

Make0{T}()              where {T} = Make{T}()
Make1{T}(x)             where {T} = Make{T}(x)
Make2{T}(x, y)          where {T} = Make{T}(x, y)
Make3{T}(x, y, z)       where {T} = Make{T}(x, y, z)
Make4{T}(x, y, z, u)    where {T} = Make{T}(x, y, z, u)
Make5{T}(x, y, z, u, v) where {T} = Make{T}(x, y, z, u, v)

# default maketype & make & Make(...)

# Make(...) is not meant to be specialized, i.e. Make(a, b, c) always create a Make3,
# and is equal to the *default* make(...)
# (it's a fall-back for client code which can help break recursivity)
# TODO: add tests for Make(...)

maketype(::Type{T}, x...) where {T} = T

Make(::Type{T}, x...) where {T} = Make{maketype(T, x...)}(x...)
make(::Type{T}, x...) where {T} = Make{maketype(T, x...)}(x...)

# make(x) is defined in sampling.jl, and is a special case wrapping already valid
# distributions (explicit or implicit)
Make(x1, x2, xs...) = Make{maketype(x1, x2, xs...)}(x1, x2, xs...)
make(x1, x2, xs...) = Make{maketype(x1, x2, xs...)}(x1, x2, xs...)

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

# show(::Make)

function Base.show(io::IO, m::Make{T}) where {T}
    M = typeof(m)
    P = M.parameters[2].parameters
    t = ntuple(length(m.x)) do i
        P[i] isa Type{<:Type} ? P[i].parameters[1] : m.x[i]
    end
    Base.show_type_name(io, M.name)
    print(io, "{", T, "}", t)
end


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

## Categorical

struct Categorical{T<:Number} <: Distribution{T}
    cdf::Vector{Float64}

    function Categorical{T}(weigths) where T
        if !isa(weigths, AbstractArray)
            # necessary for accumulate
            # TODO: will not be necessary anymore in Julia 1.5
            weigths = collect(weigths)
        end
        weigths = vec(weigths)

        isempty(weigths) &&
            throw(ArgumentError("Categorical requires at least one category"))

        s = Float64(sum(weigths))
        cdf = accumulate(weigths; init=0.0) do x, y
            x + Float64(y) / s
        end
        @assert isapprox(cdf[end], 1.0) # really?
        cdf[end] = 1.0 # to be sure the algo terminates
        new{T}(cdf)
    end
end

Categorical(weigths) = Categorical{Int}(weigths)

Categorical(n::Number) =
    Categorical{typeof(n)}(Iterators.repeated(1.0 / Float64(n), Int(n)))
