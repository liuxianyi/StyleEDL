from .GMG import GMG

models = {
    "gmg": GMG, # ours
}


def get_model(model: str):
    if model in models:
        return models[model]
    else:
        raise Exception('No such model: "%s", available: {%s}.' %
                        (model, '|'.join(models.keys())))
