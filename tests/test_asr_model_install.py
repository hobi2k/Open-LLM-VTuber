import tarfile
import tempfile
import unittest
from pathlib import Path

from open_llm_vtuber.asr.utils import check_and_extract_local_file


class ASRModelInstallTest(unittest.TestCase):
    def test_recovers_an_interrupted_extraction_from_the_downloaded_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            root = "test-model"
            source = output / "source" / root
            source.mkdir(parents=True)
            (source / "model.onnx").write_text("complete", encoding="utf-8")
            archive = output / f"{root}.tar.bz2"
            with tarfile.open(archive, "w:bz2") as tar:
                tar.add(source, arcname=root)

            extracted = output / root
            extracted.mkdir()
            (extracted / "model.onnx").write_text("partial", encoding="utf-8")

            result = check_and_extract_local_file(
                f"https://example.com/{archive.name}",
                str(output),
            )

            self.assertEqual(result, extracted)
            self.assertEqual(
                (extracted / "model.onnx").read_text(encoding="utf-8"),
                "complete",
            )
            self.assertFalse(archive.exists())
