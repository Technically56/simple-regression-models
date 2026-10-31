import hashlib, io, os, shutil, sys, tarfile, tempfile, unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))              # for `base`
sys.path.insert(0, str(REPO_ROOT / "dataset"))
import pull_dataset

URL = "https://example.invalid/data/ember_test.tar.bz2"
ARCHIVE_NAME = "ember_test.tar.bz2"
FILES = {"ember2018/train_features_0.jsonl": b'{"label": 1}\n', "ember2018/test_features.jsonl": b'{"label": 0}\n'}


def make_archive(path, files=FILES):
    """Write a .tar.bz2 at path containing files ({member name: bytes}); return its sha256."""
    with tarfile.open(path, "w:bz2") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HelpersTest(unittest.TestCase):
    def test_sha256_matches_hashlib_across_block_boundaries(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "blob"
            data = os.urandom((1 << 20) * 2 + 123)   # spans several 1 MiB read blocks
            path.write_bytes(data)
            self.assertEqual(pull_dataset._sha256(path), hashlib.sha256(data).hexdigest())

    def test_sha256_of_empty_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "empty"
            path.touch()
            self.assertEqual(pull_dataset._sha256(path), hashlib.sha256(b"").hexdigest())

    def test_progress_prints_gb_on_same_line(self):
        with mock.patch("builtins.print") as fake_print:
            pull_dataset._progress(500, 1_000_000, 4_000_000_000)
        fake_print.assert_called_once_with("\r0.50 GB / 4.00 GB", end="", flush=True)

    def test_progress_shows_unknown_when_size_missing(self):
        with mock.patch("builtins.print") as fake_print:
            pull_dataset._progress(500, 1_000_000, -1)
        fake_print.assert_called_once_with("\r0.50 GB / unknown", end="", flush=True)

    def test_project_root_is_repo_root(self):
        self.assertEqual(pull_dataset.PROJECT_ROOT, REPO_ROOT)


class DownloadEmberTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.source = self.root / "source.tar.bz2"   # what the fake server hands out
        self.sha = make_archive(self.source)
        self.dest = self.root / "out" / "ember2018"
        self.tmp = self.dest.parent / "tmp"

        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(os.environ).start()   # restored on cleanup
        for var in ("EMBER_DEST", "EMBER_URL", "EMBER_SHA256"):
            os.environ.pop(var, None)

        self.urlretrieve = mock.patch.object(pull_dataset.urllib.request, "urlretrieve", side_effect=self._fake_urlretrieve).start()
        mock.patch("builtins.print").start()

    def _fake_urlretrieve(self, url, filename, reporthook=None):
        shutil.copyfile(self.source, filename)
        if reporthook:
            size = self.source.stat().st_size
            reporthook(0, 8192, size)
            reporthook(1, 8192, size)
        return str(filename), None

    def download(self, **kwargs):
        kwargs.setdefault("dest", self.dest)
        kwargs.setdefault("url", URL)
        kwargs.setdefault("sha256", self.sha)
        return pull_dataset.download_ember(**kwargs)

    def assert_extracted(self):
        for name, data in FILES.items():
            self.assertEqual((self.dest / name.split("/", 1)[1]).read_bytes(), data)

    # --- happy path -------------------------------------------------------

    def test_downloads_verifies_and_extracts(self):
        result = self.download()

        self.assertEqual(result, self.dest)
        self.assert_extracted()
        self.urlretrieve.assert_called_once()
        self.assertEqual(self.urlretrieve.call_args.args[0], URL)

    def test_downloads_to_part_file_first(self):
        self.download()
        target = Path(self.urlretrieve.call_args.args[1])
        self.assertEqual(target.name, ARCHIVE_NAME + ".part")

    def test_cleans_up_archive_and_tmp_by_default(self):
        self.download()
        self.assertFalse(self.tmp.exists())

    def test_keep_archive_leaves_archive_in_tmp(self):
        self.download(keep_archive=True)
        self.assertEqual([p.name for p in self.tmp.iterdir()], [ARCHIVE_NAME])

    def test_tmp_kept_when_it_holds_other_files(self):
        self.tmp.mkdir(parents=True)
        (self.tmp / "unrelated.txt").write_text("keep me")
        self.download()
        self.assertEqual([p.name for p in self.tmp.iterdir()], ["unrelated.txt"])

    def test_checksum_comparison_is_case_insensitive(self):
        self.download(sha256=self.sha.upper())
        self.assert_extracted()

    # --- skipping work ----------------------------------------------------

    def test_existing_dest_returns_immediately(self):
        self.dest.mkdir(parents=True)
        result = self.download()

        self.assertEqual(result, self.dest)
        self.urlretrieve.assert_not_called()
        self.assertEqual(list(self.dest.iterdir()), [])
        self.assertFalse(self.tmp.exists())

    def test_valid_cached_archive_skips_download(self):
        self.tmp.mkdir(parents=True)
        shutil.copyfile(self.source, self.tmp / ARCHIVE_NAME)

        self.download()

        self.urlretrieve.assert_not_called()
        self.assert_extracted()

    def test_corrupt_cached_archive_is_redownloaded(self):
        self.tmp.mkdir(parents=True)
        (self.tmp / ARCHIVE_NAME).write_bytes(b"truncated")

        self.download()

        self.urlretrieve.assert_called_once()
        self.assert_extracted()

    # --- failures ---------------------------------------------------------

    def test_checksum_mismatch_raises_and_removes_part(self):
        with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
            self.download(sha256="0" * 64)

        self.assertFalse((self.tmp / (ARCHIVE_NAME + ".part")).exists())
        self.assertFalse((self.tmp / ARCHIVE_NAME).exists())
        self.assertFalse(self.dest.exists())

    def test_rejects_archive_member_escaping_extract_dir(self):
        self.sha = make_archive(self.source, {"ember2018/ok.txt": b"x", "../evil.txt": b"pwned"})

        with self.assertRaises(tarfile.FilterError):
            self.download()

        self.assertFalse((self.tmp / "evil.txt").exists())
        self.assertFalse(self.dest.exists())

    def test_archive_without_single_top_folder_raises(self):
        self.sha = make_archive(self.source, {"a/x.txt": b"x", "b/y.txt": b"y"})

        with self.assertRaises(ValueError):
            self.download()
        self.assertFalse(self.dest.exists())

    def test_leftover_staging_dir_is_cleared(self):
        stale = self.tmp / "extract" / "stale_top_folder"
        stale.mkdir(parents=True)

        self.download()

        self.assert_extracted()
        self.assertFalse(stale.exists())

    # --- configuration ----------------------------------------------------

    def test_relative_dest_resolves_against_project_root(self):
        with mock.patch.object(pull_dataset, "PROJECT_ROOT", self.root):
            result = pull_dataset.download_ember(dest="rel/ember2018", url=URL, sha256=self.sha)

        self.assertEqual(result, self.root / "rel" / "ember2018")
        self.assertTrue((result / "test_features.jsonl").exists())

    def test_env_vars_used_when_args_omitted(self):
        env = {"EMBER_DEST": str(self.dest), "EMBER_URL": URL, "EMBER_SHA256": self.sha}
        with mock.patch.dict(os.environ, env):
            result = pull_dataset.download_ember()

        self.assertEqual(result, self.dest)
        self.assertEqual(self.urlretrieve.call_args.args[0], URL)
        self.assert_extracted()

    def test_args_override_env_vars(self):
        env = {"EMBER_DEST": str(self.root / "wrong"), "EMBER_URL": "https://wrong.invalid/x.tar.bz2", "EMBER_SHA256": "0" * 64}
        with mock.patch.dict(os.environ, env):
            self.download()

        self.assertEqual(self.urlretrieve.call_args.args[0], URL)
        self.assert_extracted()
        self.assertFalse((self.root / "wrong").exists())

    def test_defaults_used_without_args_or_env(self):
        with mock.patch.object(pull_dataset, "PROJECT_ROOT", self.root), \
             mock.patch.object(pull_dataset, "DEFAULT_SHA256", self.sha):
            result = pull_dataset.download_ember()

        self.assertEqual(result, self.root / "dataset" / "ember2018")
        self.assertEqual(self.urlretrieve.call_args.args[0], pull_dataset.DEFAULT_URL)


if __name__ == "__main__":
    unittest.main()
