import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
input_path = ROOT / "THIRDPARTY.yml"
output_path = ROOT / "third_party_licenses.html"

# Load original YAML with utf-8 encoding
with input_path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

html = ['<html><head><meta charset="utf-8"><title>Third-Party Licenses</title></head><body>']
html.append(f"<h1>Third-Party Licenses for {data.get('root_name')}</h1>")

for lib in data['third_party_libraries']:
    html.append(f"<hr><h2>{lib['package_name']} {lib['package_version']}</h2>")
    html.append(f"<p><strong>License:</strong> {lib['license']}</p>")
    if 'repository' in lib:
        html.append(f"<p><strong>Repository:</strong> <a href='{lib['repository']}'>{lib['repository']}</a></p>")
    for lic in lib['licenses']:
        html.append("<pre style='white-space: pre-wrap; background:#f7f7f7; padding:1em;'>")
        html.append(lic['text'].strip())
        html.append("</pre>")

html.append("</body></html>")
output_path.write_text("\n".join(html), encoding="utf-8")
print(f"HTML license file written to: {output_path}")
