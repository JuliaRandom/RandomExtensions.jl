# generation of some containers filled with random values


function make_argument(param)
    if param isa Symbol
        param
    else
        param::Expr
        @assert param.head == :(::)
        if param.args[1] isa Symbol
            param.args[1]
        else
            @assert param.args[1].head == :curly
            @assert param.args[1].args[1] == :Type
            @assert param.args[1].args[2] isa Symbol
            param.args[1].args[2]
        end
    end
end

# arg names must not be "rng" nor "X", already in use
# (this is for nicer output of methods, to avoid cryptic names with gensym)
macro make_container(margs...)
    definitions = []
    argss = []
    for a in margs
        push!(argss,
              a isa Expr && a.head == :vect ? # optional
                [nothing, a.args[1]] :
              a isa Expr && a.head == :call && a.args[1] == :(=>) ? # curly
                [a.args[2] => a.args[3]] :
              [a])

    end
    pushfirst!(argss, [(), :X, :(::Type{X}) => :X]) # for Sampler

    for as0 in Iterators.product(argss...)
        as1 = filter(!=(nothing), collect(Any, as0))
        curlys = []
        replace!(as1) do a
            if a isa Pair
                push!(curlys, a[2])
                a.first
            else
                a
            end
        end
        for def in (:(rand(rng::AbstractRNG) where {} = rand(rng, make())),
                    :(rand() where {} = rand(GLOBAL_RNG, make())))
            append!(def.args[1].args[1].args, filter(!=(()), as1))
            as2 = copy(as1)
            as2[1], as2[2] = as2[2], as2[1]
            filter!(!=(()), as2)
            append!(def.args[2].args[2].args[3].args, map(make_argument, as2))
            append!(def.args[1].args, curlys)
            push!(definitions, def)
        end
    end
    esc(Expr(:block, definitions...))
end

@make_container(::Type{String}, [n::Integer])
# sparse vectors & matrices
# Tuple as a container
@make_container(T::Type{<:Tuple})
@make_container(::Type{Tuple}, n::Integer)
@make_container(T::Type{NTuple{N,TT} where N} => TT, n::Integer)

## arrays (same as in Random, but with explicit type specification, e.g. rand(Int, Array, 4)

macro make_array_container(Cont)
    definitions =
        [ :(rand(rng::AbstractRNG,            $Cont, dims::Dims) =                 rand(rng,        make(t, dims))),
          :(rand(                             $Cont, dims::Dims) =                 rand(GLOBAL_RNG, make(t, dims))),
          :(rand(rng::AbstractRNG,            $Cont, dims::Integer...) =           rand(rng,        make(t, Dims(dims)))),
          :(rand(                             $Cont, dims::Integer...) =           rand(GLOBAL_RNG, make(t, Dims(dims)))),

          :(rand(rng::AbstractRNG, X,         $Cont, dims::Dims) =                 rand(rng,        make(t, X, dims))),
          :(rand(                  X,         $Cont, dims::Dims) =                 rand(GLOBAL_RNG, make(t, X, dims))),
          :(rand(rng::AbstractRNG, X,         $Cont, dims::Integer...) =           rand(rng,        make(t, X, Dims(dims)))),
          :(rand(                  X,         $Cont, dims::Integer...) =           rand(GLOBAL_RNG, make(t, X, Dims(dims)))),

          :(rand(rng::AbstractRNG, ::Type{X}, $Cont, dims::Dims)       where {X} = rand(rng,        make(t, X, dims))),
          :(rand(                  ::Type{X}, $Cont, dims::Dims)       where {X} = rand(GLOBAL_RNG, make(t, X, dims))),
          :(rand(rng::AbstractRNG, ::Type{X}, $Cont, dims::Integer...) where {X} = rand(rng,        make(t, X, Dims(dims)))),
          :(rand(                  ::Type{X}, $Cont, dims::Integer...) where {X} = rand(GLOBAL_RNG, make(t, X, Dims(dims)))),
        ]
    esc(Expr(:block, definitions...))
end

@make_array_container(t::Type{<:Array})
@make_array_container(t::Type{<:BitArray})
@make_array_container(t::AbstractFloat)


## sets/dicts

function _rand!(rng::AbstractRNG, A::SetDict, n::Integer, sp::Sampler)
    empty!(A)
    while length(A) < n
        push!(A, rand(rng, sp))
    end
    A
end

rand!(                  A::SetDict, X=default_sampling(A)) = rand!(GLOBAL_RNG, A, X)
rand!(rng::AbstractRNG, A::SetDict, X=default_sampling(A)) = _rand!(rng, A, length(A), sampler(rng, X))
rand!(                  A::SetDict, ::Type{X}) where {X}  = rand!(GLOBAL_RNG, A, X)
rand!(rng::AbstractRNG, A::SetDict, ::Type{X}) where {X}  = rand!(rng, A, Sampler(rng, X))

@make_container(T::Type{<:SetDict}, n::Integer)
