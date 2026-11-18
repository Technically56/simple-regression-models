"""Run from the repository root with: streamlit run app.py."""

import json
import os
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd
import streamlit as st

from backend.math_backend import get_backend
from base import get_base_path
from benchmark import evaluate
from dataset.parse_dataset import EmberDataset
from main import create_run_dir, run_training
from training.helpers import MODEL_CLASSES, load_model, save_model
from training.train import TrainerConfig


@st.cache_resource
def cached_backend():
    return get_backend()


@st.cache_resource
def cached_splits():
    # Keep the memmaps shared; subset views are created outside the cache.
    root = Path(os.environ.get("EMBER_PROCESSED_DIR", EmberDataset.processed_dir()))
    return tuple(EmberDataset.load(name, root) for name in ("train", "val", "test"))


def main():
    st.set_page_config(page_title="EMBER regression benchmark", layout="wide")
    st.title("EMBER regression benchmark")
    st.caption(
        "Train linear or logistic regression and evaluate malware detection on the test split."
    )
    try:
        backend = cached_backend()
        train, val, test = cached_splits()
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
        st.error(str(exc))
        st.info(
            "Build the dataset from the repo root: python -m dataset.parse_dataset. "
            "For a quick build, use --limit 20000 --out-dir dataset/processed/ember2018_quick. "
            "Set EMBER_PROCESSED_DIR to that folder before launching Streamlit."
        )
        st.stop()

    with st.sidebar:
        st.header("Training settings")
        st.caption(f"Backend: {backend.name}")
        model_name = st.selectbox("Model", list(MODEL_CLASSES), index=1)
        lr = st.number_input(
            "Learning rate", min_value=0.000001, value=0.1, format="%.6f"
        )
        l2 = st.number_input(
            "L2 regularisation (lambda)", min_value=0.0, value=0.0001, format="%.6f"
        )
        epochs = st.number_input("Epochs", min_value=1, value=10, step=1)
        batch_size = st.number_input("Batch size", min_value=1, value=4096, step=1)
        patience = st.number_input("Patience", min_value=1, value=3, step=1)
        clip = st.number_input(
            "Clip features at ± (0 = off)", min_value=0.0, value=5.0, step=1.0
        )
        if len(train) > 1:
            rows = st.slider(
                "Training rows to use", 1, len(train), min(20000, len(train))
            )
        else:
            rows = len(train)
            st.caption(f"Training rows to use: {rows}")
        train_clicked = st.button("Train", type="primary")
        st.caption(
            f"Available: {len(train):,} train / {len(val):,} validation / {len(test):,} test rows"
        )

    live_chart = st.empty()
    if train_clicked:
        cfg = TrainerConfig(
            model=model_name,
            lr=lr,
            l2=l2,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            clip=clip or None,
        )
        progress = st.progress(0.0)
        records = []
        with st.status("Training…", expanded=True) as status:

            def on_epoch(record):
                records.append(
                    {
                        "epoch": record["epoch"],
                        "train_J": record["train_J"],
                        "val_J": record["val_J"],
                    }
                )
                live_chart.line_chart(pd.DataFrame(records).set_index("epoch"))
                progress.progress(record["epoch"] / cfg.epochs)
                status.update(
                    label=f"Epoch {record['epoch']} / {cfg.epochs}"
                    + (" — early stopping" if record["stopping"] else "")
                )

            try:
                train_view = train.subset(rows)
                val_view = val.subset(max(1, rows // 5))
                run_dir = create_run_dir(cfg.model)
                model, history, metrics = run_training(
                    cfg,
                    train_view,
                    val_view,
                    test,
                    backend=backend,
                    run_dir=run_dir,
                    on_epoch=on_epoch,
                )
            except (ValueError, RuntimeError, OSError) as exc:
                status.update(label="Training failed", state="error")
                st.error(str(exc))
            else:
                st.session_state["result"] = {
                    "model": model,
                    "history": history,
                    "metrics": metrics,
                    "cfg": cfg,
                    "mean": train_view.mean,
                    "inv_std": train_view.inv_std,
                    "run_dir": run_dir,
                }
                progress.progress(1.0)
                reason = (
                    "Early stopping"
                    if len(history["train_J"]) < cfg.epochs
                    else "Finished"
                )
                status.update(
                    label=f"{reason} after {len(history['train_J'])} epochs",
                    state="complete",
                )
        live_chart.empty()

    st.subheader("Saved runs")
    paths = sorted((get_base_path() / "runs").glob("*/model.npz"), reverse=True)
    chosen = st.selectbox(
        "Saved model", paths, format_func=lambda p: p.parent.name, disabled=not paths
    )
    if st.button("Load and evaluate", disabled=not paths):
        try:
            model, mean, inv_std, saved_cfg = load_model(chosen, backend)
            metrics = evaluate(
                model, backend, test, mean, inv_std, saved_cfg.batch_size, saved_cfg.clip
            )
            history_path = chosen.parent / "history.json"
            history = (
                json.loads(history_path.read_text(encoding="utf-8"))
                if history_path.exists()
                else None
            )
            st.session_state["result"] = {
                "model": model,
                "mean": mean,
                "inv_std": inv_std,
                "cfg": saved_cfg,
                "history": history,
                "metrics": metrics,
                "run_dir": chosen.parent,
            }
        except (
            ValueError,
            RuntimeError,
            OSError,
            KeyError,
            BadZipFile,
            EOFError,
        ) as exc:
            st.error(f"Could not load this run: {exc}")

    if "result" in st.session_state:
        result = st.session_state["result"]
        st.subheader("Test results")
        st.caption(
            f"{result['cfg'].model} · {result['metrics']['n']:,} test rows · {result['run_dir'].name}"
        )
        for column, name in zip(
            st.columns(4), ("accuracy", "precision", "recall", "f1")
        ):
            column.metric(
                name.upper() if name == "f1" else name.title(),
                f"{result['metrics'][name]:.2%}",
            )
        st.table(
            pd.DataFrame(
                result["metrics"]["confusion"],
                index=["Actual benign", "Actual malware"],
                columns=["Predicted benign", "Predicted malware"],
            )
        )
        if result["history"]:
            st.subheader("Cost by epoch")
            frame = pd.DataFrame(result["history"])
            frame.index = pd.RangeIndex(1, len(frame) + 1, name="epoch")
            st.line_chart(frame)
        if st.button("Save model"):
            try:
                save_model(
                    result["run_dir"] / "model.npz",
                    result["model"],
                    result["cfg"],
                    result["mean"],
                    result["inv_std"],
                )
                st.success(f"Saved {result['run_dir'] / 'model.npz'}")
            except OSError as exc:
                st.error(f"Could not save the model: {exc}")


if __name__ == "__main__":
    main()
