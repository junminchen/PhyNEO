#!/usr/bin/env python3
"""
Draw molecules from md_phyneo/mol_smiles.py mapping into a grid image.

This version supports vector output (SVG and PDF) in addition to raster PNG.
- --svg: write an SVG (vector) file
- --pdf: write a PDF (vector) file (requires cairosvg)
- --outfile: base output filename (extension is respected)
- --dpi: DPI for PNG output (default 300)
- --scale: multiplier applied to sub-image size (default 1)
- --mols-per-row: molecules per row (default 6)
- accept a comma-separated list of names or a direct SMILES test "SMILES:..."

Usage examples:
    # generate a publication-quality PDF (vector)
    python md_phyneo/draw_smiles_grid.py --pdf --outfile molecules_grid.pdf

    # generate SVG (vector)
    python md_phyneo/draw_smiles_grid.py --svg --outfile molecules_grid.svg

    # generate high-DPI PNG
    python md_phyneo/draw_smiles_grid.py --outfile molecules_grid.png --dpi 600 --scale 2

    # draw a subset by names
    python md_phyneo/draw_smiles_grid.py DMC,EMC,EC --svg --outfile subset.svg

    # test a single raw SMILES string
    python md_phyneo/draw_smiles_grid.py "SMILES:[B-]12(OC(=O)C(=O)O1)OC(=O)C(=O)O2" --svg

Requirements:
    - RDKit (conda): conda install -c conda-forge rdkit
    - pillow (for PNG output): conda/pip typically installs as dependency
    - cairosvg (pip) only if you want --pdf conversion:
        pip install cairosvg
"""
import re
import sys
import argparse
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
except Exception as e:
    raise SystemExit("RDKit is required. Install via conda: conda install -c conda-forge rdkit") from e

# cairosvg is optional and only needed for SVG->PDF conversion
try:
    import cairosvg
    _HAS_CAIROSVG = True
except Exception:
    _HAS_CAIROSVG = False

# import the mapping from the repo file
try:
    from mol_smiles import phyneo_name_mapped_smiles
except Exception as e:
    raise SystemExit("Could not import md_phyneo.mol_smiles. Run this script from the repo root and ensure the file exists.") from e


def _clean_smiles(smi: str) -> str:
    """Remove atom-mapping indices like ':1' inside brackets so RDKit can parse more reliably."""
    if smi is None:
        return ""
    cleaned = re.sub(r':\d+', '', smi)
    return cleaned.strip()


def try_parse_smiles(smi: str):
    """
    Try several RDKit parsing strategies and return (mol, error_message).
    - Normal MolFromSmiles
    - If that fails, MolFromSmiles(..., sanitize=False) then Chem.SanitizeMol to capture error details.
    """
    if not smi:
        return None, "empty SMILES"

    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return mol, None
    except Exception as e:
        normal_err = str(e)
    else:
        normal_err = "MolFromSmiles returned None"

    # Try parse without sanitize to get better diagnostics
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return None, f"MolFromSmiles(sanitize=False) returned None (prev: {normal_err})"
        try:
            Chem.SanitizeMol(mol)
            return mol, None
        except Exception as e_s:
            return None, f"SanitizeMol failed after sanitize=False parse: {e_s} (prev: {normal_err})"
    except Exception as e2:
        return None, f"MolFromSmiles(sanitize=False) also failed: {e2} (prev: {normal_err})"


def make_mols_and_legends(names_to_draw=None):
    # direct SMILES test support: "SMILES:<smi>"
    if names_to_draw and len(names_to_draw) == 1 and names_to_draw[0].upper().startswith("SMILES:"):
        raw = names_to_draw[0][7:].strip()
        smi = _clean_smiles(raw)
        mol, err = try_parse_smiles(smi)
        legends = [smi if mol is not None else f"FAILED: {smi}"]
        if mol is None:
            mol = Chem.MolFromSmiles("C")
        return [mol], legends, [(smi, err)] if err else []

    items = list(phyneo_name_mapped_smiles.items())
    if names_to_draw:
        names = [n for n in names_to_draw if n in phyneo_name_mapped_smiles]
        items = [(n, phyneo_name_mapped_smiles[n]) for n in names]
    mols = []
    legends = []
    failed = []
    for name, raw_smi in items:
        smi = _clean_smiles(raw_smi)
        mol, err = try_parse_smiles(smi)
        if mol is None:
            # fallback: try raw string as-is
            mol2, err2 = try_parse_smiles(raw_smi)
            if mol2 is not None:
                mol = mol2
                err = err2
        if mol is None:
            failed.append((name, smi if smi else raw_smi if raw_smi else "(empty)", err))
            mol = Chem.MolFromSmiles('C')  # placeholder
            legends.append(f"{name} (parse failed)")
        else:
            legends.append(name)
        mols.append(mol)
    return mols, legends, failed


def generate_svg(mols, legends, mols_per_row=6, subimg_size=(300, 220)):
    """
    Generate an SVG string of the grid using RDKit's SVG-capable grid drawer.
    """
    # RDKit's Draw.MolsToGridImage supports useSVG=True and returns SVG string
    svg = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=subimg_size, legends=legends, useSVG=True)
    # When useSVG=True, RDKit returns a string (not PIL Image)
    if isinstance(svg, bytes):
        svg = svg.decode("utf-8")
    return svg


def save_png(mols, legends, out_path, mols_per_row=6, subimg_size=(300, 220), dpi=300):
    img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=subimg_size, legends=legends)
    # Pillow Image save with DPI
    img.save(out_path, dpi=(dpi, dpi))
    return out_path


def save_svg(svg_str, out_path):
    Path(out_path).write_text(svg_str, encoding="utf-8")
    return out_path


def svg_to_pdf(svg_str, out_path):
    if not _HAS_CAIROSVG:
        raise SystemExit("cairosvg is required to convert SVG to PDF. Install with: pip install cairosvg")
    cairosvg.svg2pdf(bytestring=svg_str.encode("utf-8"), write_to=out_path)
    return out_path


def parse_cli():
    parser = argparse.ArgumentParser(description="Draw SMILES grid (SVG/PDF/PNG).")
    parser.add_argument("names", nargs="?", default=None,
                        help='Comma-separated names from mapping to draw, or "SMILES:<smiles>" to test one SMILES. If omitted, draw all.')
    parser.add_argument("--outfile", "-o", default=None, help="Output filename (extension .png/.svg/.pdf). Default: molecules_grid.(png/svg/pdf) depending on flags.")
    parser.add_argument("--svg", action="store_true", help="Write SVG output (vector).")
    parser.add_argument("--pdf", action="store_true", help="Write PDF output (vector; requires cairosvg).")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PNG output (default 300).")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for sub-image size (default 1.0).")
    parser.add_argument("--mols-per-row", type=int, default=6, help="Molecules per row.")
    return parser.parse_args()


def main():
    args = parse_cli()
    names = None
    if args.names:
        names = [s.strip() for s in args.names.split(",") if s.strip()]

    mols, legends, failed = make_mols_and_legends(names)

    # determine subimage size and out filename
    base_subimg = (300, 220)
    subimg_size = (int(base_subimg[0] * args.scale), int(base_subimg[1] * args.scale))

    # default outfile naming
    out_base = args.outfile
    if out_base:
        out_path = Path(out_base)
    else:
        # choose default depending on flags (prefer vector if requested)
        if args.pdf:
            out_path = Path("molecules_grid.pdf")
        elif args.svg:
            out_path = Path("molecules_grid.svg")
        else:
            out_path = Path("molecules_grid.png")

    printed_paths = []

    # If user requested SVG or PDF, generate SVG first (SVG is vector intermediate)
    svg_str = None
    if args.svg or args.pdf:
        svg_str = generate_svg(mols, legends, mols_per_row=args.mols_per_row, subimg_size=subimg_size)
        if args.svg:
            svg_out = out_path if out_path.suffix.lower() == ".svg" or not out_path.suffix else Path(str(out_path))
            if not svg_out.suffix:
                svg_out = svg_out.with_suffix(".svg")
            save_svg(svg_str, svg_out)
            printed_paths.append(str(svg_out.resolve()))
        if args.pdf:
            pdf_out = out_path if out_path.suffix.lower() == ".pdf" or not out_path.suffix else Path(str(out_path))
            if not pdf_out.suffix:
                pdf_out = pdf_out.with_suffix(".pdf")
            try:
                svg_to_pdf(svg_str, pdf_out)
                printed_paths.append(str(pdf_out.resolve()))
            except SystemExit:
                raise
            except Exception as e:
                raise SystemExit(f"Failed to convert SVG to PDF: {e}")

    # If neither SVG nor PDF requested, or user still wants PNG, write PNG
    if (not args.svg and not args.pdf) or out_path.suffix.lower() == ".png" or (not args.outfile and not (args.svg or args.pdf)):
        png_out = out_path if out_path.suffix.lower() == ".png" or not out_path.suffix else Path(str(out_path))
        if not png_out.suffix:
            png_out = png_out.with_suffix(".png")
        save_png(mols, legends, png_out, mols_per_row=args.mols_per_row, subimg_size=subimg_size, dpi=args.dpi)
        printed_paths.append(str(png_out.resolve()))

    print("Saved files:")
    for p in printed_paths:
        print(" -", p)

    if failed:
        print("\nWarning: some SMILES failed to parse. Details:")
        for name, smi, err in failed:
            print(f" - name: {name!r}, SMILES: {smi!r}, error: {err}")
        print("\nFor problematic SMILES (e.g. borate anions), try:")
        print(" - testing a single SMILES: python md_phyneo/draw_smiles_grid.py \"SMILES:<that-smiles>\" --svg")
        print(" - checking/simplifying the SMILES (explicit H, neutral fragment, or alternative canonical SMILES/InChI)")
        print(" - if you want I can help convert or propose corrected SMILES based on the RDKit error messages.")
    else:
        print("\nAll SMILES parsed successfully (after cleaning).")


if __name__ == "__main__":
    main()