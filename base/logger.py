import json
import logging


def setup_logging(run_dir, level=logging.INFO):
    """Send info logs to the console and run.log, and perf records to metrics.jsonl."""
    run_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for h in (logging.StreamHandler(), logging.FileHandler(run_dir / "run.log")):
        h.setFormatter(fmt)
        root.addHandler(h)

    perf_log = logging.getLogger("perf")
    for h in perf_log.handlers[:]:
        perf_log.removeHandler(h)
        h.close()
    perf_log.setLevel(logging.INFO)
    perf_log.propagate = False
    ph = logging.FileHandler(run_dir / "metrics.jsonl")
    ph.setFormatter(logging.Formatter("%(message)s"))
    perf_log.addHandler(ph)


def perf(event, **fields):
    logging.getLogger("perf").info(json.dumps({"event": event, **fields}))
