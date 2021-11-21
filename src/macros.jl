macro rand(exp)
    rand_macro(exp)
end

function rand_macro(ex)
    whereparams = []
    ex isa Expr && ex.head ∈ (:(=), :function, :->) ||
        throw(ArgumentError("@rand requires an expression defining `rand`"))
    sig = ex.args[1]
    body = ex.args[2]

    if sig.head == :where
        append!(whereparams, sig.args[2:end])
        sig = sig.args[1]
    end

    if ex.head == :function && sig.head == :tuple # anonymous function
        sig = Expr(:call, :rand, sig.args...)
    end
    if ex.head == :->
        # TODO: check that only one argument is passed
        sig = Expr(:call, :rand, sig)
    end

    sig.head == :call &&
        sig.args[1] == :rand || throw(ArgumentError(
            "@rand requires a function expression defining `rand`"))

    argname = sig.args[2].args[1] # x
    namefull = sig.args[2] # x::X
    @assert namefull.head == :(::) # TODO: throw exception

    # sub-samplers; second argument true forces Val(Inf) for the sampler
    sps = Pair{<:Any,Bool}[]

    rng = gensym()
    body = samplerize!(sps, body, argname, rng)
    istrivial = isempty(sps)

    exsig = Expr(:call,
                 :(Random.rand),
                 :($(esc(rng))::AbstractRNG),
                 esc(as_sampler(namefull, istrivial)))

    if !isempty(whereparams)
        exsig = Expr(:where, exsig, map(esc, whereparams)...)
    end

    ex = Expr(:function, exsig, esc(body))

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
    @assert sp.args[2].args[1].head == :where
    append!(sp.args[2].args[1].args, map(esc, whereparams))

    # insert x::X in the argument list, between RNG and n::Repetition
    insert!(sp.args[2].args[1].args[1].args, 3, esc(namefull))

    # insert inner samplers
    if !istrivial
        SP = [Expr(:call, :Sampler, :RNG, esc(x), many ? Val{Inf}() : :n) for (x, many) in sps]
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

# dependson checks whether an expression depends on previously defined
# variables, stored in `vars`; as a hack, `vars` also contains `:rand` as we
# can't store a sampler defined in terms of a random value
dependson(s::Symbol, vars) = s in vars
dependson(_, vars) = false
dependson(ex::Expr, vars) = any(x -> dependson(x, vars), ex.args)

# only `sps` and `vars` gets mutated, not `ex`
function samplerize!(sps, ex, name, rng; vars = Set{Symbol}([:rand]))
    samplerizeall!(args) = map(e -> samplerize!(sps, e, name, rng; vars=vars), args)

    if ex == name
        # not within a rand() call
        return Expr(:ref, name) # name -> name[]
    end
    ex isa Expr || return ex
    if ex.head == :call && ex.args[1] == :rand
        # if the rand expression depends on a local variable, don't create a sampler
        # (if this variable's value depends on a call to rand, we can't do anything,
        # but otherwise we might still create a subsampler, by splicing the code creating
        # this variable within the `Sampler` method, although the complication might
        # not be worth it)
        makesub = !any(x -> dependson(x, vars), ex.args[2:end])
        if makesub
            # we assume that if rand has more than one arg, we want
            # a Val(Inf) sampler (e.g. rand(1:9, 2, 3)
            # TODO: within a loop, we also want Val(Inf)
            push!(sps, ex.args[2] => length(ex.args) > 2)
            i = length(sps)
            Expr(:call, :rand, rng, :($name.data[$i]), samplerizeall!(ex.args[3:end])...)
            # above, call to samplerizeall! essentially to handle case `ex == name`
            # at the top of samplerize!
            # TODO: for tail arguments, allow them to contain rand calls (which
            # can create subsamplers), by assuming and @asserting they are
            # Integer
        else
            # the arguments of rand might contain other rand calls (which need
            # `rng` inserted), possibly with a subsampler
            Expr(:call, :rand, rng, samplerizeall!(ex.args[2:end])...)
        end
    else
        if ex.head == :(=) # works also for `for` loops
            if ex.args[1] isa Symbol
                push!(vars, ex.args[1])
            else
                @assert Meta.isexpr(ex.args[1], :tuple)
                push!(vars, ex.args[1].args...)
            end
            # TODO: we should probably pop! vars which come from nested blocks, i.e.
            # restore `vars` to its initial state when samplerize! was entered, except
            # keep the newly added var at this level
        end
        Expr(ex.head, samplerizeall!(ex.args)...)
    end
end
