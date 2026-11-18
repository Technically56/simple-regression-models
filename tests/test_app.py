import gc
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from test_pipeline import REPO_ROOT, write_splits


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.logging_state = []
        for logger in (logging.getLogger(), logging.getLogger("perf")):
            self.logging_state.append(
                (logger, logger.handlers[:], logger.level, logger.propagate)
            )
            logger.handlers = []

    def tearDown(self):
        st.cache_resource.clear()
        gc.collect()
        for logger, handlers, level, propagate in self.logging_state:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()
            logger.handlers = handlers
            logger.setLevel(level)
            logger.propagate = propagate

    def test_missing_dataset_shows_build_command(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                os.environ,
                {
                    "EMBER_PROCESSED_DIR": str(Path(tmp) / "missing"),
                    "MATH_BACKEND": "numpy",
                },
            ),
        ):
            st.cache_resource.clear()
            app = AppTest.from_file(str(REPO_ROOT / "app.py")).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertIn("python -m dataset.parse_dataset", app.info[0].value)

    def test_train_rerun_save_and_load(self):
        # unittest cleanups run after tearDown releases cached memmaps/log files.
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        write_splits(root / "data")
        with (
            patch.dict(
                os.environ,
                {"EMBER_PROCESSED_DIR": str(root / "data"), "MATH_BACKEND": "numpy"},
            ),
            patch("base.get_base_path", return_value=root),
            patch("main.get_base_path", return_value=root),
        ):
            st.cache_resource.clear()
            app = AppTest.from_file(str(REPO_ROOT / "app.py")).run(timeout=30)
            self.assertFalse(app.exception)
            next(w for w in app.number_input if w.label == "Epochs").set_value(3)
            next(w for w in app.button if w.label == "Train").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(len(app.metric), 4)
            self.assertEqual(app.session_state["result"]["metrics"]["n"], 24)
            expected = app.session_state["result"]["metrics"]
            next(w for w in app.number_input if w.label == "Learning rate").set_value(
                0.01
            ).run()
            self.assertEqual(app.session_state["result"]["metrics"], expected)
            next(w for w in app.button if w.label == "Save model").click().run()
            self.assertTrue(app.success)
            next(w for w in app.button if w.label == "Load and evaluate").click().run(
                timeout=30
            )
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(app.session_state["result"]["metrics"], expected)


if __name__ == "__main__":
    unittest.main()
