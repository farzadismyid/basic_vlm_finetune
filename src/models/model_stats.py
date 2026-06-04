def count_parameters(model):

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {

        "total_parameters":
            total_params,

        "trainable_parameters":
            trainable_params,
    }
