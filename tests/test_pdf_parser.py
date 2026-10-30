"""vlm_pdf_parser end to end with a fake converter (no Docling/GPU needed)."""

import json
import sys

import pytest

import vlm_pdf_parser as parser


class FakeConverter:
    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.converted = []

    def convert(self, path):
        self.converted.append(path)
        if any(path.endswith(name) for name in self.fail_on):
            raise RuntimeError("boom")
        text = f"# markdown for {path}"

        class Doc:
            def export_to_markdown(self):
                return text

        class Result:
            document = Doc()

        return Result()


@pytest.fixture
def run(monkeypatch, tmp_path):
    """run(*args, converter=...) -> (exit code, converter used)."""

    def _run(*args, converter=None):
        converter = converter or FakeConverter()
        monkeypatch.setattr(parser, "build_converter", lambda model: converter)
        monkeypatch.setattr(sys, "argv", ["vlm_pdf_parser.py", *args])
        return parser.main(), converter

    return _run


def make_pdf(path, content=b"%PDF-1.4 test"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_same_named_pdfs_in_subdirectories_get_distinct_outputs(tmp_path, run):
    """Three report.pdf files used to map to report, report_<h>, report_<h>
    -- the third overwrote the second."""
    for folder in ("a", "b", "c"):
        make_pdf(tmp_path / "in" / folder / "report.pdf", folder.encode())
    code, converter = run("--input-dir", str(tmp_path / "in"), "--output-dir", str(tmp_path / "out"), "--recursive")
    assert code == 0
    outputs = sorted(p.name for p in (tmp_path / "out").glob("*.md"))
    assert len(outputs) == 3
    manifest = json.loads((tmp_path / "out" / "_manifest.json").read_text())
    assert len({entry["out"] for entry in manifest.values()}) == 3


def test_unique_slug_never_repeats():
    taken = set()
    slugs = [parser.unique_slug("report", taken, f"dir{i}/report.pdf") for i in range(50)]
    assert len(set(slugs)) == 50


def test_unchanged_content_is_cached_even_when_mtime_changes(tmp_path, run):
    pdf = make_pdf(tmp_path / "in" / "a.pdf")
    args = ("--input-dir", str(tmp_path / "in"), "--output-dir", str(tmp_path / "out"))
    run(*args)
    import os
    os.utime(pdf, (1, 1))  # what a fresh git clone or Docker copy does to mtimes
    _code, converter = run(*args)
    assert converter.converted == []


def test_changed_content_is_reconverted(tmp_path, run):
    pdf = make_pdf(tmp_path / "in" / "a.pdf", b"one")
    args = ("--input-dir", str(tmp_path / "in"), "--output-dir", str(tmp_path / "out"))
    run(*args)
    pdf.write_bytes(b"two")
    _code, converter = run(*args)
    assert len(converter.converted) == 1


def test_legacy_manifest_is_upgraded_without_reconverting(tmp_path, run):
    pdf = make_pdf(tmp_path / "in" / "a.pdf")
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.md").write_text("# old")
    (out / "_manifest.json").write_text(json.dumps({"a.pdf": {"key": parser.legacy_cache_key(pdf), "out": "a.md"}}))
    _code, converter = run("--input-dir", str(tmp_path / "in"), "--output-dir", str(out))
    assert converter.converted == []
    assert json.loads((out / "_manifest.json").read_text())["a.pdf"]["key"] == parser.cache_key(pdf)


def test_corrupt_manifest_is_set_aside_and_existing_outputs_are_not_overwritten(tmp_path, run):
    make_pdf(tmp_path / "in" / "a.pdf")
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.md").write_text("# converted earlier from another folder's a.pdf")
    (out / "_manifest.json").write_text('{"a.pdf": {"key": "x", "out"')  # truncated write
    code, _converter = run("--input-dir", str(tmp_path / "in"), "--output-dir", str(out))
    assert code == 0
    assert (out / "_manifest.json.corrupt").exists()
    assert (out / "a.md").read_text().startswith("# converted earlier")  # reserved, not overwritten
    assert len(list(out.glob("a_*.md"))) == 1


def test_exit_codes(tmp_path, run):
    assert run("--input-dir", str(tmp_path / "missing"))[0] == 2
    make_pdf(tmp_path / "in" / "good.pdf", b"good")
    make_pdf(tmp_path / "in" / "bad.pdf", b"bad")
    code, _converter = run(
        "--input-dir", str(tmp_path / "in"), "--output-dir", str(tmp_path / "out"),
        converter=FakeConverter(fail_on={"bad.pdf"}),
    )
    assert code == 1
    assert (tmp_path / "out" / "good.md").exists()


def test_list_only_converts_nothing(tmp_path, run):
    make_pdf(tmp_path / "in" / "a.PDF")
    code, converter = run("--input-dir", str(tmp_path / "in"), "--list")
    assert code == 0 and converter.converted == []


def test_changing_the_model_reconverts_and_records_it(tmp_path, run):
    make_pdf(tmp_path / "in" / "a.pdf")
    out = tmp_path / "out"
    args = ("--input-dir", str(tmp_path / "in"), "--output-dir", str(out))
    run(*args, "--model", "SMOLDOCLING_TRANSFORMERS")
    _code, converter = run(*args, "--model", "SMOLDOCLING_TRANSFORMERS")
    assert converter.converted == []  # same model: cached
    _code, converter = run(*args)  # default model: different Markdown, so reconvert
    assert len(converter.converted) == 1
    assert json.loads((out / "_manifest.json").read_text())["a.pdf"]["model"] == parser.DEFAULT_MODEL


def test_entry_without_a_recorded_model_counts_as_the_default(tmp_path, run):
    pdf = make_pdf(tmp_path / "in" / "a.pdf")
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.md").write_text("# old")
    (out / "_manifest.json").write_text(json.dumps({"a.pdf": {"key": parser.cache_key(pdf), "out": "a.md"}}))
    args = ("--input-dir", str(tmp_path / "in"), "--output-dir", str(out))
    _code, converter = run(*args)
    assert converter.converted == []
    _code, converter = run(*args, "--model", "SMOLDOCLING_TRANSFORMERS")
    assert len(converter.converted) == 1
