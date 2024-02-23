import json

if __name__ == "__main__":
    folders = {
        "out-enwik8-inter": "+ intermediate loss",
        "out-enwik8-inter-convnorm21": "+ intermediate loss + convnorm (L=21)",
        "out-enwik8-inter-selfpred": "+ intermediate loss + next layer pred.",
        "out-enwik8-inter-convnorm11": "+ intermediate loss + convnorm (L=11)",
        "out-enwik8-inter-selfcond": "+ intermediate loss + self-conditioning",
        "out-enwik8-inter-convnorm11-shared": "+ intermediate loss + convnorm (L=11, shared)",
        "out-enwik8-inter-predict1": "+ intermediate loss + 1 extra target",
        "out-enwik8-inter-predict1-selfcond": "+ intermediate loss + 1 extra target + self-conditioning",
        "out-enwik8": "baseline",
    }
    model_sizes = {
        "out-enwik8": 56.79,
        "out-enwik8-selfcond": 56.95,
        "out-enwik8-selfpred": 11.14,
        "out-enwik8-selfcond-predict": 57.58,
        "out-enwik8-predict": 57.11,
        "out-enwik8-selfcond-inter-predict1": 56.95,
        "out-enwik8-inter-predict1": 56.95,
        "out-enwik8-inter-predict1-selfcond": 57.27,
        "out-enwik8-inter-selfcond": 56.95,
        "out-enwik8-inter": 56.79,
        "out-enwik8-inter-convnorm11": 57.06,
        "out-enwik8-inter-convnorm21": 57.31,
        "out-enwik8-inter-convnorm3": 56.87,
        "out-enwik8-inter-selfpred": 59.16,
        "out-enwik8-inter-convnorm11-shared": 56.79,
        "out-enwik8-inter-selfcond-v2": 56.95,
    }

    results = {}
    for exp_folder in folders:
        with open(exp_folder + "/test.json") as f:
            results[exp_folder] = json.load(f)["bpc"]

    order_by_bpc = sorted(folders, key=lambda x: results[x], reverse=True)

    longest_string = max(len(folders[exp_folder]) for exp_folder in order_by_bpc)

    fmt = f"{{:{longest_string}s}} {{:.2f}}M   {{:.3f}} bpc"
    for exp_folder in order_by_bpc:
        print(
            fmt.format(
                folders[exp_folder], model_sizes[exp_folder], results[exp_folder]
            )
        )
