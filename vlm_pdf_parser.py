"""
Generic bulk Docling conversion of PDFs to Markdown.

Uses Docling's VLM pipeline with the granite-docling-258M model
on GPU (CUDA) if available, falling back to CPU.
Model weights cache to ~/.cache/huggingface by default (override with
--model-cache-dir) so the download only happens once, independent of
--output-dir.
Converted markdown caches to the output directory along with a _manifest.json,
keyed on (filename, size, mtime) -- a source PDF that hasn't changed is never
re-run. Note: this cache key is not a content hash, so mtimes lost across a
fresh git clone/Docker copy will look "changed" on the first run there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def unique_slug(stem: str, taken: set[str]) -> str:
    """Slugify, disambiguating collisions (e.g. 'a.b' and 'a_b' both -> 'a_b')."""
    base = slugify(stem)
    if base not in taken:
        taken.add(base)
        return base
    suffix = hashlib.sha1(stem.encode()).hexdigest()[:8]
    candidate = f"{base}_{suffix}"
    taken.add(candidate)
    return candidate


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict, manifest_path: Path) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def cache_key(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.name}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def find_pdfs(input_dir: Path, recursive: bool) -> list[Path]:
    """Case-insensitive match on *.pdf/*.PDF/etc, optionally recursive, deduped."""
    pattern = "**/*" if recursive else "*"
    seen = {}
    for p in input_dir.glob(pattern):
        if p.is_file() and p.suffix.lower() == ".pdf":
            seen[p.resolve()] = p
    return sorted(seen.values())


def build_converter(model_spec_name: str):
    """VLM pipeline on CUDA if available, else CPU. model_spec_name is an
    attribute name on docling.datamodel.vlm_model_specs, e.g. GRANITEDOCLING_TRANSFORMERS."""
    import torch
    from docling.datamodel import vlm_model_specs
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    try:
        vlm_spec = getattr(vlm_model_specs, model_spec_name)
    except AttributeError:
        raise SystemExit(
            f"Unknown VLM model spec {model_spec_name!r}. "
            f"Check docling.datamodel.vlm_model_specs for valid names."
        )

    device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
    print(f"Docling VLM pipeline device: {device} "
          f"(cuda available: {torch.cuda.is_available()})", file=sys.stderr)

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_spec,
        accelerator_options=AcceleratorOptions(device=device),
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )


def main():
    ap = argparse.ArgumentParser(description="Convert PDFs to Markdown using Docling's VLM pipeline.")
    ap.add_argument("--input-dir", type=str, default=".", help="Directory to scan for PDFs (default: current directory)")
    ap.add_argument("--output-dir", type=str, default="./parsed_markdown", help="Directory to save markdown files (default: ./parsed_markdown)")
    ap.add_argument("--recursive", action="store_true", help="Also search subdirectories of --input-dir")
    ap.add_argument("--model-cache-dir", type=str, default=None,
                     help="Where to cache downloaded VLM weights (default: HF_HOME env var, or ~/.cache/huggingface)")
    ap.add_argument("--model", type=str, default="GRANITEDOCLING_TRANSFORMERS",
                     help="Attribute name in docling.datamodel.vlm_model_specs to use (default: GRANITEDOCLING_TRANSFORMERS)")
    ap.add_argument("--force", action="store_true", help="Re-convert even if a fresh cache entry exists")
    ap.add_argument("--list", action="store_true", help="Only list matching PDFs, don't convert")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / "_manifest.json"

    if args.model_cache_dir:
        os.environ["HF_HOME"] = str(Path(args.model_cache_dir).resolve())
    else:
        os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return

    targets = find_pdfs(input_dir, args.recursive)

    if not targets:
        print(f"No PDFs found in {input_dir}{' (recursive)' if args.recursive else ''}.")
        return

    print(f"Found {len(targets)} PDFs in {input_dir}:")
    for p in targets:
        print(f"  {p.relative_to(input_dir)}")

    if args.list:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)

    taken_slugs = {entry["out"][:-3] for entry in manifest.values() if "out" in entry}
    to_convert = []
    for p in targets:
        key = cache_key(p)
        manifest_key = str(p.relative_to(input_dir))
        existing = manifest.get(manifest_key, {})
        if existing.get("out"):
            out_path = output_dir / existing["out"]
        else:
            out_path = output_dir / f"{unique_slug(p.stem, taken_slugs)}.md"
        if not args.force and existing.get("key") == key and out_path.exists():
            continue
        to_convert.append((p, manifest_key, key, out_path))

    print(f"\n{len(targets) - len(to_convert)} already cached and unchanged, "
          f"{len(to_convert)} need conversion.")

    if not to_convert:
        return

    converter = build_converter(args.model)

    for i, (p, manifest_key, key, out_path) in enumerate(to_convert, 1):
        t0 = time.time()
        print(f"[{i}/{len(to_convert)}] Converting {manifest_key} ...", file=sys.stderr)
        try:
            result = converter.convert(str(p))
            md = result.document.export_to_markdown()
        except Exception:
            print(f"  FAILED:\n{traceback.format_exc()}", file=sys.stderr)
            continue

        out_path.write_text(md, encoding="utf-8")
        elapsed = time.time() - t0
        manifest[manifest_key] = {"key": key, "out": out_path.name, "chars": len(md), "seconds": round(elapsed, 1)}
        save_manifest(manifest, manifest_path)
        print(f"  -> {out_path.name} ({len(md)} chars, {elapsed:.1f}s)", file=sys.stderr)

    print(f"\nDone. Markdown cached under {output_dir}/, manifest at {manifest_path}")


if __name__ == "__main__":
    main()
