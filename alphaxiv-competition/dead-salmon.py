import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Probing Experiment: ModernBERT Layer-wise Language Classification

    This notebook tests whenever a language model learns to distinguish between different languages, by training classifiers to identify the language of sampled sentences based on the model's internal representations and comparing their accuracy an untrained model.
    """)
    return


@app.cell(hide_code=True)
def _(DATASET_NAME, MODEL_NAME, SUBSET_NAME, mo):
    mo.md(f"""Experiment configuration:
    - Model: `{MODEL_NAME}`
    - Dataset: `{DATASET_NAME}` (Subset: `{SUBSET_NAME}`)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Proccess

    First, we load the dataset and select the parts we care about (the sentence pairs as well as the language of each part)
    """)
    return


@app.cell
def _(DATASET_NAME, SUBSET_NAME, load_dataset):
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split="train").to_polars()
    dataset
    return (dataset,)


@app.cell
def _(dataset, pl):
    sentences = pl.concat([
        dataset.select(pl.col("src").alias("sentence"), pl.col("sl").alias("language")),
        dataset.select(pl.col("trg").alias("sentence"), pl.col("tl").alias("language")),
    ])
    sentences
    return (sentences,)


@app.cell
def _(dataset):
    del dataset
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we run them all through the model and extract the activations of each layer, once for the trained model, once for the untrained model.
    """)
    return


@app.cell
def _(extract_activations, sentences, tokenizer, trained_model):
    trained_activations = extract_activations(trained_model, tokenizer, sentences)
    return (trained_activations,)


@app.cell
def _(trained_model):
    del trained_model
    return


@app.cell
def _(extract_activations, sentences, tokenizer, untrained_model):
    untrained_activations = extract_activations(untrained_model, tokenizer, sentences)
    return (untrained_activations,)


@app.cell
def _(untrained_model):
    del untrained_model
    return


@app.cell(hide_code=True)
def _(mo, trained_activations):
    mo.md(f"""The activations are dictionaries of layer_idx -> numpy array [N sentences, hidden size]
    - Number of layers = number of items in the dictionaries: `{len(trained_activations) = }`
    - Number of sentences = first dimension of each array: `{trained_activations[0].shape[0] = }`
    - Hidden size = second dimension of each array: `{trained_activations[0].shape[1] = }`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next up, we train small Classifier models trying to predict the language from the extracted Activations to measure their accuracy.
    """)
    return


@app.cell
def _(pl, sentences):
    binary_labels = sentences.select(pl.col("language").eq(pl.col("language").first()))['language'].to_numpy()
    return (binary_labels,)


@app.cell
def _(binary_labels, probe_all_layers, trained_activations):
    trained_res = probe_all_layers(trained_activations, binary_labels)
    return (trained_res,)


@app.cell
def _(trained_activations,):
    del trained_activations
    return


@app.cell
def _(binary_labels, probe_all_layers, untrained_activations):
    untrained_res = probe_all_layers(untrained_activations, binary_labels)
    return (untrained_res,)


@app.cell
def _(untrained_activations):
    del untrained_activations
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results
    """)
    return


@app.cell
def _(np, trained_res, untrained_res):
    trained_accuracy = np.array([res[1][0] for res in sorted(trained_res.items())])
    untrained_accuracy = np.array([res[1][0] for res in sorted(untrained_res.items())])
    return trained_accuracy, untrained_accuracy


@app.cell
def _(np, trained_res, untrained_res):
    trained_std = np.array([res[1][1] for res in sorted(trained_res.items())])
    untrained_std = np.array([res[1][1] for res in sorted(untrained_res.items())])
    return trained_std, untrained_std


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, let's check the accuracy of the probes made from the trained model's Activations
    """)
    return


@app.cell
def _(px, trained_accuracy, trained_std):
    px.line(trained_accuracy, range_y=(0.0, 1.0), error_y=trained_std)
    return


@app.cell(hide_code=True)
def _(mo, np, trained_accuracy):
    mo.md(fr"""
    Almost perfect accross the board!
    Even the worst layer still got a score of {np.min(trained_accuracy).item():%}

    However, it is still early to conclude that the ModernBERT component is what learned that.


    Let's check the untrained model next:
    """)
    return


@app.cell
def _(px, untrained_accuracy, untrained_std):
    px.line(untrained_accuracy, range_y=(0.0, 1.0), error_y=untrained_std)
    return


@app.cell(hide_code=True)
def _(mo, np, untrained_accuracy):
    mo.md(fr"""
    Almost perfect accross the board?!


    Notice how despite the ModernBERT part being untrained, the worst layer still got a score of {np.min(untrained_accuracy).item():%}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    Despite the trained ModernBERT model getting good results on the tests, we cannot necessarily state that it has learned to distinguish languages based on our experiment.

    While there are multiple hypothesis we could come up with to explain why the untrained model activations can still be used to distinguish between languages, to draw conclusions from experiments we must not only ensure that they work under the circumstances we expect them to work under, we must also test whenever they fail outside of these circumstances.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Utilities
    """)
    return


@app.function
def mean_pool(hidden_state, attention_mask):
    """Average non-padding token representations."""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


@app.cell
def _(BATCH_SIZE, MAX_LENGTH, np, pl, torch):
    def extract_activations(model, tokenizer, sentences):
        """
        Returns a dict  layer_idx → numpy array  [N, hidden_size]
        Includes the embedding layer (index 0) plus all transformer layers.
        """
        n_layers = model.config.num_hidden_layers   # transformer blocks only
        # +1 for the initial embedding layer → indices 0 … n_layers
        all_reps = {i: [] for i in range(n_layers + 1)}
 
        for start in range(0, len(sentences), BATCH_SIZE):
            batch_texts = sentences.select(pl.col("sentence").slice(start, BATCH_SIZE))['sentence'].to_list()
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
 
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
 
            # out.hidden_states: tuple of (n_layers+1) tensors, each [B, T, H]
            for layer_idx, hs in enumerate(out.hidden_states):
                pooled = mean_pool(hs, enc["attention_mask"])
                all_reps[layer_idx].append(pooled.float().numpy())
 
        return {k: np.concatenate(v, axis=0) for k, v in all_reps.items()}

    return (extract_activations,)


@app.cell
def _(
    LogisticRegression,
    N_FOLDS,
    SEED,
    StandardScaler,
    StratifiedKFold,
    accuracy_score,
    np,
):
    def probe_layer(X, y):
        """Stratified k-fold logistic regression probe. Returns mean accuracy and standard deviation."""
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        accs = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
 
            scaler = StandardScaler()
            X_tr  = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
 
            clf = LogisticRegression(random_state=SEED)
            clf.fit(X_tr, y_tr)
            accs.append(accuracy_score(y_val, clf.predict(X_val)))
 
        return float(np.mean(accs)), float(np.std(accs))

    return (probe_layer,)


@app.cell
def _(probe_layer):
    def probe_all_layers(activations, labels):
        """Run probing for every layer. Returns a dict of layer_idx => (mean_acc, std_acc)."""
        results = {}
        for layer_idx in sorted(activations.keys()):
            mean_acc, std_acc = probe_layer(activations[layer_idx], labels)
            results[layer_idx] = (mean_acc, std_acc)
        return results

    return (probe_all_layers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Config
    """)
    return


@app.cell
def _():
    SEED = 42
    return (SEED,)


@app.cell
def _():
    MODEL_NAME = "answerdotai/ModernBERT-base"
    return (MODEL_NAME,)


@app.cell
def _():
    DATASET_NAME = "google/smol"
    SUBSET_NAME = "smolsent__en_es"
    return DATASET_NAME, SUBSET_NAME


@app.cell
def _():
    N_FOLDS = 10
    return (N_FOLDS,)


@app.cell
def _():
    BATCH_SIZE = 32
    return (BATCH_SIZE,)


@app.cell
def _():
    MAX_LENGTH = 256
    return (MAX_LENGTH,)


@app.cell
def _(AutoTokenizer, MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return (tokenizer,)


@app.cell
def _(AutoModel, MODEL_NAME):
    trained_model = AutoModel.from_pretrained(MODEL_NAME)
    trained_model.eval()
    return (trained_model,)


@app.cell
def _(AutoConfig, AutoModel, MODEL_NAME):
    _config = AutoConfig.from_pretrained(MODEL_NAME)
    untrained_model = AutoModel.from_config(_config)
    untrained_model.eval()
    return (untrained_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dependencies
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import time

    return


@app.cell
def _():
    import random

    return


@app.cell
def _():
    import numpy as np

    return (np,)


@app.cell
def _():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    return LogisticRegression, StandardScaler, StratifiedKFold, accuracy_score


@app.cell
def _():
    import torch

    return (torch,)


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _():
    import plotly.express as px

    return (px,)


@app.cell
def _():
    from datasets import load_dataset

    return (load_dataset,)


@app.cell
def _():
    from transformers import AutoTokenizer, AutoModel, AutoConfig

    return AutoConfig, AutoModel, AutoTokenizer


if __name__ == "__main__":
    app.run()
