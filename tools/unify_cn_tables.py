"""Unify CN table widths with resizebox matching EN v45."""
import re

filepath = r"e:\pytorchFile\YSC_2\paper\TWC_CN_34.tex"
with open(filepath, "r", encoding="utf-8-sig") as f:
    content = f.read()

# Identify each \begin{tabular}...\end{tabular} block and wrap with resizebox
# Regex: setlength line + tabular. We match the tabular block and size.
# Determine width by surrounding table context: table* => \textwidth, table => \columnwidth

def wrap_callback(match):
    full = match.group(0)
    # decide width by whether preceding context is table* (we pass a flag via lookbehind)
    return full

# Simpler: process sequentially with a state for table* vs table
# Split by \begin{table / \begin{table* and handle each block
result = []
idx = 0
pattern = re.compile(r'\\begin\{table\*?\}(\[[^\]]*\])?')
pos = 0
while True:
    m = pattern.search(content, pos)
    if not m:
        result.append(content[pos:])
        break
    result.append(content[pos:m.start()])
    # determine if it's table* (wide) or table
    is_wide = m.group(0).startswith('\\begin{table*')
    width = '\\textwidth' if is_wide else '\\columnwidth'
    # find the end of this table env
    endm = re.search(r'\\end\{table\*?\}', content[m.end():])
    body = content[m.start():m.end()+endm.end()]
    # within body, wrap first \begin{tabular} ... \end{tabular}
    tabm = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}', body, re.DOTALL)
    if tabm:
        tabblock = tabm.group(0)
        wrapped = '\\resizebox{%s}{!}{%%\n%s}' % (width, tabblock)
        body = body[:tabm.start()] + wrapped + body[tabm.end():]
    result.append(body)
    pos = m.end() + endm.end()

new_content = ''.join(result)

with open(filepath, "w", encoding="utf-8-sig") as f:
    f.write(new_content)

print("CN table width unification complete.")
print("resizebox count:", new_content.count("\\resizebox"))
print("table env count:", len(re.findall(r'\\begin\{table\*?\}', new_content)))