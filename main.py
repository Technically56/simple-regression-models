from base.base_path import get_base_path
from dataset import download_ember
from dataset.parse_dataset import EmberDataset
from dataset.pull_dataset import download_ember


def prepare_data(limit_rows: int | None = None):
    if not EmberDataset.is_built():
        raw = get_base_path() / "dataset" / "ember2018"
        if not (raw / "test_features.jsonl").exists():
            download_ember()
        EmberDataset.build(seed=0)

    train = EmberDataset.load("train")
    val = EmberDataset.load("val")
    if limit_rows:
        train, val = train.subset(limit_rows), val.subset(limit_rows // 5)
    return train, val


def main():
    train, val = prepare_data()
    model = LogisticRegressionModel(backend, train.n_features, l2=cfg.l2)
    history = Trainer(model, backend, train, val, cfg).train()
