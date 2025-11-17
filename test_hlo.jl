using Reactant

# Add your project
using Pkg
Pkg.activate(".")

include("src/KAEM/KAEM.jl")
using .KAEM_model

# Set up a minimal test case
using ConfParser
conf = ConfParse("config/cifar_config.ini")
parse_conf!(conf)

# Initialize model with small sizes
model = init_KAEM_model(conf)

# Create small test inputs
z_test = Reactant.to_rarray(randn(Float32, 2, 8))  # Small batch

# Get parameters and states
ps, st_kan, st_lux = initialize_states(model)

# Try to see the HLO without compiling
println("Generating HLO code...")
@time Reactant.@code_hlo model.log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm)
