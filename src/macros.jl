macro rand(exp)
    rand_macro(exp)
end

function rand_macro(ex)
    ex isa Expr && ex.head ∈ (:(=), :function) ||
        throw(ArgumentError("@rand requires an expression defining `rand`"))
    sig = ex.args[1]
    body = ex.args[2]

    sig.head == :call &&
        sig.args[1] == :rand || throw(ArgumentError(
            "@rand requires a function expression defining `rand`"))

    argname = sig.args[2].args[1] # x
    namefull = sig.args[2] # x::X
    @assert namefull.head == :(::) # TODO: throw exception

    sps = Any[] # sub-samplers
    rng = gensym()
    body = samplerize!(sps, body, argname, rng)
    istrivial = isempty(sps)
    # rand -> Base.rand

    exsig = Expr(:call,
                 :(Random.rand),
                 :($(esc(rng))::AbstractRNG),
                 esc(as_sampler(namefull, istrivial)))

    ex = Expr(ex.head, exsig, esc(body))

    sp = if istrivial
        # we explicitly define Sampler even in the trivial case to handle
        # redefinitions, where the old rand/sampler pair (for SamplerSimple)
        # is overwritten by a new one (for SamplerTrivial)
        quote
            Random.Sampler(::Type{RNG}, n::Repetition) where {RNG<:AbstractRNG} =
                SamplerTrivial($(esc(argname)))
        end
    else
        quote
            Random.Sampler(::Type{RNG}, n::Repetition) where {RNG<:AbstractRNG} =
                SamplerSimple($(esc(argname)), tuple(SP))
        end
    end

    # insert x::X in the argument list, between RNG and n::Repetition
    insert!(sp.args[2].args[1].args[1].args, 3, esc(namefull))

    # insert inner samplers
    if !istrivial
        SP = [Expr(:call, :Sampler, :RNG, esc(x), :n) for x in sps]
        @assert :SP == pop!(sp.args[2].args[2].args[2].args[3].args)
        append!(sp.args[2].args[2].args[2].args[3].args, SP)
    end

    quote
        $ex
        $(sp.args[2]) # unwrap the quote/block around the definition
    end
end

function as_sampler(ex, istrivial)
    t = istrivial ? :(RandomExtensions.SamplerTrivial) : :(RandomExtensions.SamplerSimple)
    Expr(:(::),
         ex.args[1],
         Expr(:curly, t,
              Expr(:(<:), ex.args[2])))
end

function samplerize!(sps, ex, name, rng)
    if ex == name
        # not within a rand() call
        return Expr(:ref, name) # name -> name[]
    end
    ex isa Expr || return ex
    if ex.head == :call && ex.args[1] == :rand
        # TODO: handle Repetition == Val(Inf) for arrays
        push!(sps, ex.args[2])
        i = length(sps)
        Expr(:call, :rand, rng, :($name.data[$i]), ex.args[3:end]...)
    else
        Expr(ex.head, map(e -> samplerize!(sps, e, name, rng), ex.args)...)
    end
end
