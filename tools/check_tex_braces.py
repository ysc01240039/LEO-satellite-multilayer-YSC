"""Check LaTeX file for brace/environment mismatches"""
import re, sys

fpath = sys.argv[1] if len(sys.argv) > 1 else 'paper/YSC_EN.tex'
with open(fpath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Check environment balance
begins = sum(line.count('\\begin{') for line in lines)
ends = sum(line.count('\\end{') for line in lines)
print(f"Total \\begin: {begins}, Total \\end: {ends}")

# Find environment stack
envs = []
for i, line in enumerate(lines, 1):
    for m in re.finditer(r'\\begin\{(\w+\*?)\}', line):
        envs.append(('begin', m.group(1), i))
    for m in re.finditer(r'\\end\{(\w+\*?)\}', line):
        envs.append(('end', m.group(1), i))

stack = []
for op, env, lineno in envs:
    if op == 'begin':
        stack.append((env, lineno))
    else:
        if stack and stack[-1][0] == env:
            stack.pop()
        else:
            print(f"MISMATCH at line {lineno}: \\end{{{env}}} but expected {stack[-1] if stack else 'nothing'}")
if stack:
    print(f"UNCLOSED environments: {stack}")

# Check overall brace balance
text = ''.join(lines)
opens = text.count('{')
closes = text.count('}')
print(f"\nTotal braces: {{ = {opens}, }} = {closes}, diff = {opens - closes}")
if opens != closes:
    # Find the position where braces become unbalanced
    balance = 0
    for i, ch in enumerate(text):
        if ch == '{':
            balance += 1
        elif ch == '}':
            balance -= 1
        if balance < 0:
            # Find the line
            lineno = text[:i].count('\n') + 1
            print(f"First excess }} at position {i}, around line {lineno}")
            break
    if balance > 0:
        lineno = text.count('\n') + 1
        print(f"Missing {balance} closing braces, near end of file (line {lineno})")