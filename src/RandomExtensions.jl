module RandomExtensions

export Combine, Uniform, Normal, Exponential, CloseOpen

import Random: Sampler, rand

using Random
using Random: SamplerTrivial, SamplerSimple, SamplerTag, Repetition

include("distributions.jl")
include("sampling.jl")

end # module
