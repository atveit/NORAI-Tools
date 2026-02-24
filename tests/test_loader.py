"""Tests for norai_tools.loader."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from norai_tools.loader import load_checkpoint, save_checkpoint, save_improved


class TestLoadCheckpoint:
    def test_no_checkpoint_returns_empty(self, tmp_path):
        rows, idx = load_checkpoint(str(tmp_path / "nonexistent.jsonl"))
        assert rows == []
        assert idx == 0

    def test_load_existing_checkpoint(self, tmp_path):
        cp_path = tmp_path / "checkpoint.jsonl"
        data = [
            {"instruction": "a", "output": "b"},
            {"instruction": "c", "output": "d"},
        ]
        with open(cp_path, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        rows, idx = load_checkpoint(str(cp_path))
        assert len(rows) == 2
        assert idx == 2
        assert rows[0]["instruction"] == "a"
        assert rows[1]["instruction"] == "c"

    def test_load_checkpoint_skips_empty_lines(self, tmp_path):
        cp_path = tmp_path / "checkpoint.jsonl"
        with open(cp_path, "w") as f:
            f.write('{"a": 1}\n\n{"b": 2}\n')

        rows, idx = load_checkpoint(str(cp_path))
        assert len(rows) == 2
        assert idx == 2

    def test_load_checkpoint_default_path(self):
        """Test that default path uses CHECKPOINT_FILE constant."""
        with patch("norai_tools.loader.Path") as mock_path_cls:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = False
            mock_path_cls.return_value = mock_path_instance
            rows, idx = load_checkpoint()
            assert rows == []
            assert idx == 0


class TestSaveCheckpoint:
    def test_save_creates_file(self, tmp_path):
        cp_path = tmp_path / "checkpoint.jsonl"
        save_checkpoint([{"a": 1}], str(cp_path))
        assert cp_path.exists()

    def test_save_appends_to_file(self, tmp_path):
        cp_path = tmp_path / "checkpoint.jsonl"
        save_checkpoint([{"a": 1}], str(cp_path))
        save_checkpoint([{"b": 2}], str(cp_path))

        with open(cp_path) as f:
            lines = [l for l in f.readlines() if l.strip()]
        assert len(lines) == 2

    def test_save_valid_jsonl(self, tmp_path):
        cp_path = tmp_path / "checkpoint.jsonl"
        data = [{"instruction": "Skriv noe", "output": "Her er noe"}]
        save_checkpoint(data, str(cp_path))

        with open(cp_path) as f:
            loaded = json.loads(f.readline())
        assert loaded["instruction"] == "Skriv noe"

    def test_save_handles_unicode(self, tmp_path):
        cp_path = tmp_path / "checkpoint.jsonl"
        data = [{"text": "Blåbær og rømme med ærlig smak"}]
        save_checkpoint(data, str(cp_path))

        with open(cp_path, encoding="utf-8") as f:
            loaded = json.loads(f.readline())
        assert "Blåbær" in loaded["text"]
        assert "ærlig" in loaded["text"]


class TestSaveImproved:
    def test_save_requires_hub_repo_for_push(self):
        from datasets import Dataset
        ds = Dataset.from_list([{"a": 1}])
        with pytest.raises(ValueError, match="hub_repo"):
            save_improved(ds, "/tmp/test.parquet", push_to_hub=True)

    def test_save_to_parquet(self, tmp_path):
        from datasets import Dataset
        ds = Dataset.from_list([{"col1": "val1", "col2": "val2"}])
        out_path = str(tmp_path / "output.parquet")
        save_improved(ds, out_path)
        assert Path(out_path).exists()
