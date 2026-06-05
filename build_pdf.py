"""Convierte el README (PRD) a PDF con estilo de reporte.

Uso: python build_pdf.py
Genera: PRD_Pricing_Dinamico_Retail.pdf
"""

from __future__ import annotations

import re
from pathlib import Path

import markdown
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "README.md"
OUT = ROOT / "PRD_Pricing_Dinamico_Retail.pdf"

# Fuentes Windows: Arial (texto, soporta acentos) y Consolas (monoespaciada,
# soporta caracteres de dibujo de caja └ ─ │ ▼ usados en los diagramas).
ARIAL = "C:/Windows/Fonts/arial.ttf"
ARIAL_BD = "C:/Windows/Fonts/arialbd.ttf"
CONSOLAS = "C:/Windows/Fonts/consola.ttf"

# reportlab (motor de xhtml2pdf) no renderiza emoji a color; se sustituyen por
# texto para que el semáforo de calidad se lea correctamente en el PDF.
EMOJI = {
    "🔴": "[ROJO]",
    "🟡": "[AMARILLO]",
    "🟢": "[VERDE]",
    "📊": "",
    "└": "└", "─": "─", "│": "│", "├": "├", "┬": "┬", "┐": "┐", "▼": "▼", "▲": "▲",
}

CSS = """
@page {
    size: a4 portrait;
    margin: 2.0cm 1.8cm 2.0cm 1.8cm;
    @frame footer { -pdf-frame-content: footerContent; bottom: 1.0cm; margin-left: 1.8cm; margin-right: 1.8cm; height: 1cm; }
}
body { font-family: "Body"; font-size: 9.5pt; color: #1f1f1f; line-height: 1.45; }
h1 { font-size: 20pt; color: #1C1C1C; border-bottom: 3px solid #C49A00; padding-bottom: 6px; margin-top: 4px; }
h2 { font-size: 14.5pt; color: #1C1C1C; border-bottom: 1px solid #C49A00; padding-bottom: 3px; margin-top: 18px; }
h3 { font-size: 11.5pt; color: #8a6d00; margin-top: 14px; }
h4 { font-size: 10pt; color: #444; margin-top: 10px; }
p, li { font-size: 9.5pt; }
blockquote { color: #555; border-left: 3px solid #C49A00; padding: 4px 10px; background: #faf6ea; }
code { font-family: "Mono"; font-size: 8.5pt; background: #f1f1f1; color: #8a3b00; padding: 0 2px; }
pre { font-family: "Mono"; font-size: 7.6pt; background: #f6f6f6; border: 1px solid #ddd;
      padding: 8px; line-height: 1.25; }
pre code { background: transparent; color: #222; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; }
th { background: #1C1C1C; color: #F5C518; font-size: 8.5pt; padding: 5px 6px; text-align: left; }
td { border: 1px solid #ccc; font-size: 8.5pt; padding: 4px 6px; vertical-align: top; }
tr:nth-child(even) td { background: #f7f7f7; }
hr { border: none; border-top: 1px solid #ddd; margin: 14px 0; }
a { color: #8a6d00; text-decoration: none; }
"""


def _register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Body", ARIAL))
    pdfmetrics.registerFont(TTFont("Body-Bold", ARIAL_BD))
    pdfmetrics.registerFont(TTFont("Mono", CONSOLAS))
    # Mapea negritas dentro de la familia "Body" para <b>/<strong>/headings.
    addMapping("Body", 0, 0, "Body")
    addMapping("Body", 1, 0, "Body-Bold")
    addMapping("Body", 0, 1, "Body")
    addMapping("Body", 1, 1, "Body-Bold")


def main() -> None:
    _register_fonts()
    text = SRC.read_text(encoding="utf-8")
    for emo, repl in EMOJI.items():
        if repl != emo:
            text = text.replace(emo, repl)

    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
    )

    full = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
<div id="footerContent" style="font-size:7.5pt;color:#999;text-align:center;">
  PRD — Pricing Dinámico Retail · Office Max México · página <pdf:pagenumber> de <pdf:pagecount>
</div>
{html_body}
</body></html>"""

    with open(OUT, "wb") as fh:
        result = pisa.CreatePDF(full, dest=fh, encoding="utf-8")

    if result.err:
        raise SystemExit(f"Errores al generar PDF: {result.err}")
    print(f"PDF generado: {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
