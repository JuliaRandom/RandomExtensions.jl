macro rand(exp)
    rand_macro(exp)
end

function rand_macro(ex)
    ex isa Expr &&
        ex.head ∈ (:(=), :function) &&
        ex.args[1].head == :call &&
        ex.args[1].args[1] == :rand || throw(ArgumentError(
            "@rand requires a function expression defining `rand`"))

    name = ex.args[1].args[2].args[1] # x
    namefull = ex.args[1].args[2] # x::X
    @assert namefull.head == :(::)
    namefull.args[2] = esc(namefull.args[2])

    sps = Any[] # sub-samplers
    body = samplerize!(sps, ex.args[2], name)
    istrivial = isempty(sps)
    # rand -> Base.rand
    ex = Expr(ex.head, Expr(:call, :(Random.rand),
                            :(rng::AbstractRNG),
                            as_sampler(ex.args[1].args[2], istrivial)),
              body)
    istrivial && return ex
    sp = quote
        Random.Sampler(::Type{RNG}, n::Repetition) where {RNG<:AbstractRNG} =
            SamplerSimple($name, tuple(SP))
    end
    # insert x::X in the argument list, between RNG and n::Repetition
    insert!(sp.args[2].args[1].args[1].args, 3, namefull)
    SP = [Expr(:call, :Sampler, :RNG, x, :n) for x in sps]
    @assert :SP == pop!(sp.args[2].args[2].args[2].args[3].args)
    append!(sp.args[2].args[2].args[2].args[3].args, SP)
    quote
        $ex
        $(sp.args[2]) # unwrap the quote/block around the definition
    end
end

function as_sampler(ex, istrivial)
    t = istrivial ? :SamplerTrivial : :SamplerSimple
    Expr(:(::),
         ex.args[1],
         Expr(:curly, t,
              Expr(:(<:), ex.args[2])))
end

function samplerize!(sps, ex, name)
    ex isa Expr || return ex
    if ex.head == :call && ex.args[1] == :rand
        # TODO: handle Repetition == Val(Inf) for arrays
        push!(sps, ex.args[2])
        i = length(sps)
        Expr(:call, :rand, :rng, :($name.data[$i]), ex.args[3:end]...)
    else
        Expr(ex.head, map(e -> samplerize!(sps, e, name), ex.args)...)
    end
end
