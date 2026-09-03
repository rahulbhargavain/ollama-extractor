"""
Generic bulk Docling conversion of PDFs to Markdown.

Uses Docling's VLM pipeline with the granite-docling-258M model
on GPU (CUDA) if available, falling back to CPU.
Model weights cache to .docling_models/ so the download only happens once.
Converted markdown caches to the output directory along with a _manifest.json,
keyed on (filename, size, mtime) -- a source PDF that hasn't changed is never
re-run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

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

def build_converter():
    """VLM pipeline (granite-docling-258M) on CUDA if available, else CPU."""
    import torch
    from docling.datamodel import vlm_model_specs
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
    print(f"Docling VLM pipeline device: {device} "
          f"(cuda available: {torch.cuda.is_available()})", file=sys.stderr)

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
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
    ap.add_argument("--force", action="store_true", help="Re-convert even if a fresh cache entry exists")
    ap.add_argument("--list", action="store_true", help="Only list matching PDFs, don't convert")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    models_dir = output_dir / ".docling_models"
    manifest_path = output_dir / "_manifest.json"

    os.environ.setdefault("HF_HOME", str(models_dir))

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return

    targets = sorted(input_dir.glob("*.pdf"))
    
    if not targets:
        print(f"No PDFs found in {input_dir}.")
        return
        
    print(f"Found {len(targets)} PDFs in {input_dir}:")
    for p in targets:
        print(f"  {p.name}")
        
    if args.list:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)

    to_convert = []
    for p in targets:
        key = cache_key(p)
        out_path = output_dir / f"{slugify(p.stem)}.md"
        if not args.force and manifest.get(p.name, {}).get("key") == key and out_path.exists():
            continue
        to_convert.append((p, key, out_path))

    print(f"\n{len(targets) - len(to_convert)} already cached and unchanged, "
          f"{len(to_convert)} need conversion.")
    
    if not to_convert:
        return

    converter = build_converter()

    for i, (p, key, out_path) in enumerate(to_convert, 1):
        t0 = time.time()
        print(f"[{i}/{len(to_convert)}] Converting {p.name} ...", file=sys.stderr)
        try:
            result = converter.convert(str(p))
            md = result.document.export_to_markdown()
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            continue
            
        out_path.write_text(md, encoding="utf-8")
        elapsed = time.time() - t0
        manifest[p.name] = {"key": key, "out": out_path.name, "chars": len(md), "seconds": round(elapsed, 1)}
        save_manifest(manifest, manifest_path)
        print(f"  -> {out_path.name} ({len(md)} chars, {elapsed:.1f}s)", file=sys.stderr)

    print(f"\nDone. Markdown cached under {output_dir}/, manifest at {manifest_path}")

if __name__ == "__main__":
    main()
