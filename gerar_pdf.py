"""Converte o README.md em relatorio.pdf, com estilo limpo e imagens embutidas."""
import markdown
import base64
import os
import re
import subprocess

with open("README.md", "r", encoding="utf-8") as f:
    md_text = f.read()

# Converte markdown -> HTML (com suporte a tabelas e blocos de codigo)
html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "sane_lists"]
)

# Embute as imagens como base64 para o PDF nao depender de caminhos externos
def embed_image(match):
    src = match.group(1)
    if src.startswith("http"):
        return match.group(0)
    if os.path.exists(src):
        with open(src, "rb") as img:
            b64 = base64.b64encode(img.read()).decode()
        return f'src="data:image/png;base64,{b64}"'
    return match.group(0)

html_body = re.sub(r'src="([^"]+)"', embed_image, html_body)

# A primeira linha do README (cabecalho obrigatorio) vira um paragrafo destacado
html = f"""<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 12px;
          line-height: 1.5; color: #222; margin: 32px; }}
  h1 {{ font-size: 22px; border-bottom: 2px solid #4F6FA8; padding-bottom: 6px; }}
  h2 {{ font-size: 17px; color: #2f4368; margin-top: 24px;
        border-bottom: 1px solid #ccc; padding-bottom: 3px; }}
  h3 {{ font-size: 14px; color: #3a3a3a; margin-top: 16px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #bbb; padding: 6px 9px; text-align: left; font-size: 11px; }}
  th {{ background: #4F6FA8; color: #fff; }}
  tr:nth-child(even) {{ background: #f4f6fa; }}
  code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px;
          font-family: 'DejaVu Sans Mono', monospace; font-size: 11px; }}
  pre {{ background: #f5f5f5; padding: 10px; border-radius: 5px;
         border: 1px solid #ddd; overflow-x: auto; }}
  pre code {{ background: none; padding: 0; }}
  img {{ max-width: 100%; height: auto; display: block; margin: 10px auto; }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 18px 0; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

with open("_readme.html", "w", encoding="utf-8") as f:
    f.write(html)

subprocess.run([
    "wkhtmltopdf", "--enable-local-file-access",
    "--margin-top", "14mm", "--margin-bottom", "14mm",
    "--margin-left", "12mm", "--margin-right", "12mm",
    "_readme.html", "relatorio.pdf"
], check=True)

os.remove("_readme.html")
print("relatorio.pdf gerado com sucesso.")
