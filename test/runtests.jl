using RandomExtensions, Random, SparseArrays
using Test

@testset "Distributions" begin
    # Normal/Exponential
    @test rand(Normal()) isa Float64
    @test rand(Normal(0.0, 1.0)) isa Float64
    @test rand(Exponential()) isa Float64
    @test rand(Exponential(1.0)) isa Float64
    @test rand(Normal(Float32)) isa Float32
    @test rand(Exponential(Float32)) isa Float32
    @test rand(Normal(ComplexF64)) isa ComplexF64

    # pairs/complexes
    @test rand(Combine(Pair, 1:3, Float64)) isa Pair{Int,Float64}
    @test rand(Pair{Int,Float64}) isa Pair{Int,Float64}
    z = rand(Combine(Complex, 1:3, 6:9))
    @test z.re ∈ 1:3
    @test z.im ∈ 6:9
    @test z isa Complex{Int}
    z = rand(Combine(ComplexF64, 1:3, 6:9))
    @test z.re ∈ 1:3
    @test z.im ∈ 6:9
    @test z isa ComplexF64
    for (C, R) in ((Complex, Int), (ComplexF64, Float64), (Complex{Int}, Int))
        z = rand(Combine(C, 1:3))
        @test z.re ∈ 1:3
        @test z.im ∈ 1:3
        @test z isa Complex{R}
    end
    @test rand(ComplexF64) isa ComplexF64

    @test rand(Combine(Complex,Int), 3) isa Vector{Complex{Int}}
    @test rand(Combine(Complex,1:3), 3) isa Vector{Complex{Int}}

    # Uniform
    @test rand(Uniform(Float64)) isa Float64
    @test rand(Uniform(1:10)) isa Int
    @test rand(Uniform(1:10)) ∈ 1:10
    @test rand(Uniform(Int)) isa Int
end

@testset "Containers" for rng in ([], [MersenneTwister(0)], [RandomDevice()])
    # Set
    for s = (rand(rng..., 1:99, Set{Int}, 10),
             rand(rng..., 1:99, Set, 10))
        @test s isa Set{Int}
        @test length(s) == 10
        @test rand(s) ∈ 1:99
    end
    s = Set([1, 2])
    @test s === rand!(s)
    @test first(s) ∉ (1, 2) # extremely unlikely
    @test length(s) == 2
    @test s === rand!(s, 3:9) <= Set(3:9)
    @test length(s) == 2

    @test rand(rng..., Pair{Int,Float64}, Set, 3) isa Set{Pair{Int,Float64}}
    @test rand(rng..., Pair{Int,Float64}, Set{Pair}, 3) isa Set{Pair}

    # Dict
    for s = (rand(rng..., Combine(Pair, 1:99, 1:99), Dict, 10),
             rand(rng..., Combine(Pair, 1:99, 1:99), Dict{Int,Int}, 10))
        @test s isa Dict{Int,Int}
        @test length(s) == 10
        p = rand(s)
        @test p.first ∈ 1:99
        @test p.second ∈ 1:99
    end
    s = Dict(1=>2, 2=>1)
    @test s === rand!(s)
    @test length(s) == 2
    @test first(s).first ∉ (1, 2) # extremely unlikely
    rand!(s, Combine(Pair, 3:9, Int))
    @test length(s) == 2
    @test first(s).first ∈ 3:9

    d = rand(rng..., Pair{Int,Float64}, Dict, 3)
    @test d isa Dict{Int,Float64}
    dd = rand!(rng..., d, Pair{Int,Int8})
    @test dd === d
    delt = pop!(d)
    @test delt isa Pair{Int,Float64}
    @test delt[2] ∈ typemin(Int8):typemax(Int8)
    @test rand(rng..., Pair{Int,Float64}, Dict{Any,Any}, 3) isa Dict{Any,Any}

    # sparse
    @test rand(rng..., Float64, .5, 10) isa SparseVector{Float64}
    @test rand(rng..., .5, 10) isa SparseVector{Float64}
    @test rand(rng..., Int, .5, 10) isa SparseVector{Int}
    @test rand(rng..., Float64, .5, 10, 3) isa SparseMatrixCSC{Float64}
    @test rand(rng..., .5, 10, 3) isa SparseMatrixCSC{Float64}
    @test rand(rng..., Int, .5, 10, 3) isa SparseMatrixCSC{Int}

    # BitArray
    @test rand(rng..., BitArray, 10) isa BitVector
    @test rand(rng..., BitVector, 10) isa BitVector
    @test_throws MethodError rand(rng..., BitVector, 10, 20) isa BitVector
    @test rand(rng..., BitArray, 10, 3) isa BitMatrix
    @test rand(rng..., BitMatrix, 10, 3) isa BitMatrix
    @test_throws MethodError rand(rng..., BitVector, 10, 3) isa BitMatrix

    # String
    s = rand(rng..., String)
    @test s isa String
    @test length(s) == 8
    s = rand(rng..., String, 10)
    @test s isa String
    @test length(s) == 10
    s = rand(rng..., "asd", String)
    @test length(s) == 8
    @test Set(s) <= Set("asd")
end

@testset "Rand" for rng in ([], [MersenneTwister(0)], [RandomDevice()])
    for XT = zip(([Int], [1:3], []), (Int, Int, Float64))
        X, T = XT
        r = Rand(rng..., X...)
        @test collect(Iterators.take(r, 10)) isa Vector{T}
        @test r() isa T
        @test r(2, 3) isa Matrix{T}
        @test r(.3, 2, 3) isa SparseMatrixCSC{T}
    end
    for d = (Uniform(1:10), Uniform(Int))
        @test collect(Iterators.take(d, 10)) isa Vector{Int}
    end
end

struct PairDistrib <: RandomExtensions.Distribution{Pair}
end

Random.rand(rng::AbstractRNG, ::Random.SamplerTrivial{PairDistrib}) = 1=>2

@testset "allow abstract Pair when generating a Dict" begin
    d = rand(PairDistrib(), Dict, 1)
    @test d == Dict(1=>2)
    @test typeof(d) == Dict{Any,Any}
end

@testset "some tight typing" begin
    UI = Random.UInt52()
    @test eltype(rand(MersenneTwister(), Random.Sampler(MersenneTwister, UI), .6, 1, 0)) == UInt64
    @test eltype(rand(UI, Set, 3)) == UInt64
    @test eltype(rand(Uniform(UI), 3)) == UInt64
    a = rand(RandomExtensions.Combine(Pair, Int, UI))
    @test fieldtype(typeof(a), 2) == UInt64
end

@testset "rand(::Pair)" begin
    @test rand(1=>3) ∈ (1, 3)
    @test rand(1=>2, 3) isa Vector{Int}
    @test rand(1=>'2', 3) isa Vector{Union{Char, Int}}
end
