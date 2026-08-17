import fitz
import sys

doc = fitz.open(r'e:\pytorchFile\YSC_2\results\YSC-新改-乔.pdf')
print(f"Total pages: {doc.page_count}")
print("=" * 80)

for i in range(doc.page_count):
    page = doc[i]
    annots = page.annots()
    if annots:
        for a in annots:
            info = a.info
            subtype = info.get("subtype", "")
            content = info.get("content", "")
            title = info.get("title", "")
            if content:
                print(f"--- Page {i+1} | Type: {subtype} | Author: {title} ---")
                print(content)
                print()

# Also extract text content for context
print("=" * 80)
print("TEXT CONTENT (first 15 pages)")
print("=" * 80)
for i in range(min(15, doc.page_count)):
    page = doc[i]
    text = page.get_text()
    if text.strip():
        print(f"\n--- Page {i+1} ---")
        print(text[:2000])

doc.close()