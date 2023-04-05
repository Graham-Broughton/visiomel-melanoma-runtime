import itertools as it

import apex
import efficientnet_pytorch


def get_model(CFG):
    return efficientnet_pytorch.EfficientNet.from_pretrained(
        CFG.name,
        num_classes=CFG.num_classes,
    )


def add_weight_decay(model, weight_decay=1e-4, skip_list=("bn",)):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def get_opt(CFG, model, criterion):
    # Scale learning rate based on global batch size
    if CFG.opt == "Adam":
        opt = apex.optimizers.FusedAdam(
            it.chain(
                model.parameters(), criterion.parameters()
            ),  # add_weight_decay(model, CFG.weight_decay, ('bn', )),
            lr=CFG.lr,
            weight_decay=CFG.weight_decay,
        )
    elif CFG.opt == "SGD":
        opt = apex.optimizers.FusedSGD(
            add_weight_decay(model, CFG.weight_decay, ("bn",)),
            CFG.lr,
            momentum=CFG.momentum,
            weight_decay=CFG.weight_decay,
        )
    else:
        raise

    return opt