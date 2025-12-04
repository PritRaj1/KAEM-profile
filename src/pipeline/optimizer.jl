module optimization

export opt, create_opt

using Lux, ConfParser, Optimisers

struct opt
    rule
end

function create_opt(conf::ConfParse)
    """
    Create an optimizer from a configuration file.

    Args:
        conf: ConfParse object

    Returns:
        opt: opt object, which initializes the optimizer when called
    """

    LR = parse(Float32, retrieve(conf, "OPTIMIZER", "learning_rate"))
    β = parse.(Float32, retrieve(conf, "OPTIMIZER", "betas")) |> Tuple
    decay = parse(Float32, retrieve(conf, "OPTIMIZER", "decay"))
    ρ = parse(Float32, retrieve(conf, "OPTIMIZER", "ρ"))
    ε = parse(Float32, retrieve(conf, "OPTIMIZER", "ε"))

    opt_type = retrieve(conf, "OPTIMIZER", "type")

    opt_mapping = Dict(
        # "adam" => () -> Optimisers.Adam(LR, β, ε),
        # "adamw" => () -> Optimisers.AdamW(LR, β, decay, ε),
        "momentum" => () -> Optimisers.Momentum(LR, ρ),
        "nesterov" => () -> Optimisers.Nesterov(LR, ρ),
        "sgd" => () -> Optimisers.Descent(LR),
        "rmsprop" => () -> Optimisers.RMSProp(LR, ρ, ε)
    )

    return opt(get(opt_mapping, opt_type, () -> Optimisers.Adam(LR, β, ε)))
end

end
