module trainer

export KAEM_trainer, init_trainer, train!

using Flux: onecold, mse
using Random, ComponentArrays, CSV, HDF5, JLD2, ConfParser, Reactant
using Optimization, OptimizationOptimJL, Lux, LinearAlgebra, Accessors
using MultivariateStats: reconstruct
using MLDataDevices: cpu_device

include("../utils.jl")
using .Utils

include("../KAEM/KAEM.jl")
using .KAEM_model

include("../KAEM/model_setup.jl")
using .ModelSetup

include("../KAEM/grid_updating.jl")
using .ModelGridUpdating

include("optimizer.jl")
include("data_utils.jl")
using .optimization
using .DataUtils: get_vision_dataset, get_text_dataset

mutable struct KAEM_trainer{T <: Float32}
    model::Any
    cnn::Bool
    o::opt
    dataset_name::AbstractString
    ps::ComponentArray{T}
    st_kan::ComponentArray{T}
    st_lux::NamedTuple
    N_epochs::Int
    train_loader_state::Tuple{Any, Int}
    x::AbstractArray{T}
    num_generated_samples::Int
    batch_size_for_gen::Int
    grid_update_frequency::Int
    last_grid_update::Int
    save_model::Bool
    gen_type::AbstractString
    checkpoint_every::Int
    gen_every::Int
    loss::T
    rng::AbstractRNG
end

function init_trainer(
        rng::AbstractRNG,
        conf::ConfParse,
        dataset_name;
        img_resize = nothing,
        file_loc = nothing,
        save_model = true,
    )

    N_train = parse(Int, retrieve(conf, "TRAINING", "N_train"))
    N_test = parse(Int, retrieve(conf, "TRAINING", "N_test"))
    num_generated_samples = parse(Int, retrieve(conf, "TRAINING", "num_generated_samples"))
    batch_size_for_gen = parse(Int, retrieve(conf, "TRAINING", "batch_size_for_gen"))
    seq = dataset_name == "PTB" || dataset_name == "SMS_SPAM"
    gen_type = seq ? "logits" : "images"
    cnn = parse(Bool, retrieve(conf, "CNN", "use_cnn_lkhood"))
    sequence_length = seq ? parse(Int, retrieve(conf, "SEQ", "sequence_length")) : 0
    commit!(conf, "SEQ", "sequence_length", string(sequence_length)) # Make sure 0 is set if not sequence
    vocab_size = parse(Int, retrieve(conf, "SEQ", "vocab_size"))
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    dataset, x_shape, save_dataset = (
        seq ?
            get_text_dataset(
                dataset_name,
                N_train,
                N_test,
                num_generated_samples;
                sequence_length = sequence_length,
                vocab_size = vocab_size,
                batch_size = batch_size,
            ) :
            get_vision_dataset(
                dataset_name,
                N_train,
                N_test,
                num_generated_samples;
                img_resize = img_resize,
                cnn = cnn,
            )
    )

    println("Dataset loaded")

    # Log against ULA and autoMALA
    N_t = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps"))
    mala =
        parse(Bool, retrieve(conf, "POST_LANGEVIN", "use_langevin")) ? "autoMALA" :
        "importance"

    if mala == "autoMALA" && !parse(Bool, retrieve(conf, "POST_LANGEVIN", "use_autoMALA"))
        mala = "ULA"
    end


    model_type =
        N_t > 1 ? "Thermodynamic/$(dataset_name)/$(mala)" :
        "Vanilla/$(dataset_name)/$(mala)"

    prior_spline_fcn =
        retrieve(conf, "EbmModel", "π_0") *
        "_" *
        retrieve(conf, "EbmModel", "spline_function")
    if mala == "importance"
        model_type = model_type * "/" * prior_spline_fcn
    end

    if length(parse.(Int, retrieve(conf, "EbmModel", "layer_widths"))) > 2
        model_type = model_type * "/deep"
    elseif parse(Bool, retrieve(conf, "MixtureModel", "use_mixture_prior"))
        model_type = model_type * "/mixture"
    else
        model_type = model_type * "/univariate"
    end

    file_loc = isnothing(file_loc) ? "logs/$(model_type)/" : file_loc
    mkpath(file_loc)

    println("Initializing model...")
    model = init_KAEM(dataset, conf, x_shape; file_loc = file_loc, rng = rng)
    x, loader_state = iterate(model.train_loader)
    x = pu(x)
    model, params, st_kan, st_lux = prep_model(model, x; rng = rng)

    optimizer = create_opt(conf)
    grid_update_frequency =
        parse(Int, retrieve(conf, "GRID_UPDATING", "grid_update_frequency"))

    N_epochs = parse(Int, retrieve(conf, "TRAINING", "N_epochs"))
    checkpoint_every = parse(Int, retrieve(conf, "TRAINING", "checkpoint_every"))
    gen_every = parse(Int, retrieve(conf, "TRAINING", "gen_every"))

    try
        h5write(file_loc * "real_$(gen_type).h5", "samples", save_dataset)
    catch
        rm(file_loc * "real_$(gen_type).h5")
        h5write(file_loc * "real_$(gen_type).h5", "samples", save_dataset)
    end

    open(file_loc * "loss.csv", "w") do file
        write(file, "Time (s),Epoch,Train MLE Loss,Test MSE Loss,Grid Updated\n")
    end

    return KAEM_trainer(
        model,
        cnn,
        optimizer,
        dataset_name,
        params,
        st_kan,
        st_lux,
        N_epochs,
        loader_state,
        x,
        num_generated_samples,
        batch_size_for_gen,
        grid_update_frequency,
        1,
        save_model,
        gen_type,
        checkpoint_every,
        gen_every,
        zero(Float32),
        rng,
    )
end

function train!(t::KAEM_trainer; train_idx::Int = 1)

    num_batches = length(t.model.train_loader)
    grid_updated = 0
    num_param_updates = num_batches * t.N_epochs

    loss_file = t.model.file_loc * "loss.csv"

    grid_compiled = Reactant.@compile update_model_grid(
        t.model,
        t.x,
        t.ps,
        t.st_kan,
        Lux.testmode(t.st_lux),
        train_idx,
        t.rng
    )

    gen_compiled = Reactant.@compile t.model(
        t.ps,
        t.st_kan,
        Lux.testmode(t.st_lux),
        size(t.x)[end],
        t.rng,
    )

    # Gradient for a single batch
    function grad_fcn(G, u, args...)
        t.ps = u

        # Grid updating for likelihood model
        if (
                train_idx == 1 || (train_idx - t.last_grid_update >= t.grid_update_frequency)
            ) && (t.model.update_llhood_grid || t.model.update_prior_grid)
            t.ps, t.st_kan, t.st_lux = grid_compiled(
                t.model,
                t.x,
                t.ps,
                t.st_kan,
                Lux.testmode(t.st_lux),
                train_idx,
                t.rng,
            )

            t.grid_update_frequency =
                train_idx > 1 ?
                floor(t.grid_update_frequency * (2 - t.model.grid_update_decay)^train_idx) :
                t.grid_update_frequency
            t.last_grid_update = train_idx
            grid_updated = 1

            t.model.verbose && println("Iter: $(train_idx), Grid updated")
        end

        # Rand like this cannot be compiled with MLIR
        swap_replica_idxs = (
            t.model.N_t > 1 ?
                rand(t.rng, 1:(t.model.N_t - 1), t.model.posterior_sampler.N) :
                nothing
        )

        t.loss, grads, st_ebm, st_gen = t.model.loss_fcn(
            t.ps,
            t.st_kan,
            t.st_lux,
            t.model,
            t.x,
            train_idx,
            t.rng,
            swap_replica_idxs
        )
        copy!(G, (grads))
        @reset t.st_lux.ebm = st_ebm
        @reset t.st_lux.gen = st_gen

        t.model.verbose && println("Iter: $(train_idx), Loss: $(t.loss)")
        return G
    end

    train_loss = 0

    # Train and test loss with logging
    function opt_loss(u, args...)
        t.ps = u
        train_loss += t.loss

        # After one epoch, calculate test loss and log to CSV
        if train_idx % num_batches == 0 || train_idx == 1

            test_loss = 0
            for x in t.model.test_loader
                x_gen, st_ebm, st_gen = gen_compiled(
                    t.model,
                    t.ps,
                    t.st_kan,
                    Lux.testmode(t.st_lux),
                    size(x)[end],
                    t.rng,
                )
                @reset t.st_lux.ebm = st_ebm
                @reset t.st_lux.gen = st_gen

                # MSE loss between pixels for images, and max index for logits
                if t.gen_type == "logits"
                    idxs = dropdims(argmax(x_gen, dims = 1); dims = 1)
                    test_loss +=
                        sum((pu(onecold(x, 1:size(x, 1))) .- getindex.(idxs, 1)) .^ 2) /
                        size(x)[end]
                else
                    test_loss += mse(pu(x), x_gen)
                end
            end

            train_loss = train_loss / num_batches
            test_loss /= length(t.model.test_loader)
            now_time = time() - start_time
            epoch = train_idx == 1 ? 0 : fld(train_idx, num_batches)

            t.model.verbose && println(
                "Epoch: $(epoch), Train Loss: $(train_loss), Test Loss: $(test_loss)",
            )

            open(loss_file, "a") do file
                write(file, "$now_time,$(epoch),$train_loss,$test_loss,$grid_updated\n")
            end

            (t.checkpoint_every > 0 && epoch % t.checkpoint_every) == 0 && jldsave(
                t.model.file_loc * "ckpt_epoch_$(epoch).jld2";
                params = t.ps |> cpu_device(),
                kan_state = t.st_kan |> cpu_device(),
                lux_state = t.st_lux |> cpu_device(),
                rng = t.rng,
            )

            train_loss = 0
            grid_updated = 0

            # Save images - collect batches first then concatenate once to avoid O(n²) allocations
            num_batches_to_save = fld(t.num_generated_samples, 10) ÷ t.batch_size_for_gen # Save 1/10 of the samples to conserve space
            if num_batches_to_save > 0
                concat_dim = length(t.model.lkhood.x_shape) + 1

                # Get first batch to determine type
                first_batch, st_ebm, st_gen = gen_compiled(
                    t.ps,
                    t.st_kan,
                    Lux.testmode(t.st_lux),
                    t.batch_size_for_gen,
                    t.rng,
                )
                @reset t.st_lux.ebm = st_ebm
                @reset t.st_lux.gen = st_gen
                batches_to_cat = Vector{typeof(cpu_device()(first_batch))}()
                sizehint!(batches_to_cat, num_batches_to_save)
                push!(batches_to_cat, cpu_device()(first_batch))

                for i in 2:num_batches_to_save
                    batch, st_ebm, st_gen = gen_compiled(
                        t.ps,
                        t.st_kan,
                        Lux.testmode(t.st_lux),
                        t.batch_size_for_gen,
                        t.rng,
                    )
                    @reset t.st_lux.ebm = st_ebm
                    @reset t.st_lux.gen = st_gen
                    push!(batches_to_cat, cpu_device()(batch))
                end
                gen_data = cat(batches_to_cat..., dims = concat_dim)
            else
                gen_data = zeros(Float32, t.model.lkhood.x_shape..., 0)
            end

            if (t.gen_every > 0 && epoch % t.gen_every == 0)

                if !t.model.lkhood.SEQ && !t.model.lkhood.CNN && t.model.use_pca
                    gen_data = reconstruct(t.model.PCA_model, gen_data)
                    gen_data = (
                        reshape(
                            gen_data,
                            t.model.original_data_size...,
                            size(gen_data)[end],
                        ),
                    )
                end

                try
                    h5write(
                        t.model.file_loc * "generated_$(t.gen_type)_epoch_$(epoch).h5",
                        "samples",
                        gen_data,
                    )
                catch
                    rm(t.model.file_loc * "generated_$(t.gen_type)_epoch_$(epoch).h5")
                    h5write(
                        t.model.file_loc * "generated_$(t.gen_type)_epoch_$(epoch).h5",
                        "samples",
                        gen_data,
                    )
                end
            end

        end

        train_idx += 1

        # Iterate loader, reset to first batch when epoch ends
        x, t.train_loader_state =
            (train_idx % num_batches == 0) ? iterate(t.model.train_loader) :
            iterate(t.model.train_loader, t.train_loader_state)
        t.x = pu(x)

        return t.loss
    end

    start_time = time()

    optf = Optimization.OptimizationFunction(opt_loss; grad = grad_fcn)
    optprob = Optimization.OptimizationProblem(optf, t.ps)

    # Optimization only stops when maxiters is reached
    res = Optimization.solve(
        optprob,
        t.o.init_optimizer();
        maxiters = num_param_updates,
        verbose = true,
        abstol = -one(Float32),
        reltol = -one(Float32),
        x_tol = -one(Float32),
        x_abstol = -one(Float32),
        x_reltol = -one(Float32),
        f_tol = -one(Float32),
        f_abstol = -one(Float32),
        f_reltol = -one(Float32),
        g_tol = -one(Float32),
        g_abstol = -one(Float32),
        g_reltol = -one(Float32),
        outer_x_abstol = -one(Float32),
        outer_x_reltol = -one(Float32),
        outer_f_abstol = -one(Float32),
        outer_f_reltol = -one(Float32),
        outer_g_abstol = -one(Float32),
        outer_g_reltol = -one(Float32),
        successive_f_tol = num_param_updates,
        allow_f_increases = true,
        allow_outer_f_increases = true,
    )

    # Generate samples
    num_batches = t.num_generated_samples ÷ t.batch_size_for_gen
    concat_dim = length(t.model.lkhood.x_shape) + 1
    first_batch, st_ebm, st_gen = gen_compiled(
        t.ps,
        t.st_kan,
        Lux.testmode(t.st_lux),
        t.batch_size_for_gen,
        t.rng,
    )
    batches_to_cat = Vector{typeof(cpu_device()(first_batch))}()
    sizehint!(batches_to_cat, num_batches)
    push!(batches_to_cat, cpu_device()(first_batch))

    for i in 2:num_batches
        batch, st_ebm, st_gen = gen_compiled(
            t.ps,
            t.st_kan,
            Lux.testmode(t.st_lux),
            t.batch_size_for_gen,
            t.rng,
        )
        push!(batches_to_cat, cpu_device()(batch))
    end
    gen_data = cat(batches_to_cat..., dims = concat_dim)

    if !t.model.lkhood.SEQ && !t.model.lkhood.CNN && t.model.use_pca
        gen_data = reconstruct(t.model.PCA_model, gen_data)
        gen_data =
            (reshape(gen_data, t.model.original_data_size..., size(gen_data)[end]))
    end

    try
        h5write(t.model.file_loc * "generated_$(t.gen_type).h5", "samples", gen_data)
    catch
        rm(t.model.file_loc * "generated_$(t.gen_type).h5")
        h5write(t.model.file_loc * "generated_$(t.gen_type).h5", "samples", gen_data)
    end

    return t.save_model && jldsave(
        t.model.file_loc * "saved_model.jld2";
        params = t.ps |> cpu_device(),
        kan_state = t.st_kan |> cpu_device(),
        lux_state = t.st_lux |> cpu_device(),
        train_idx = train_idx,
    )
end

end
