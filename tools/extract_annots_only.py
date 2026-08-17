import fitz

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
            content = info.get("content", "").strip()
            title = info.get("title", "")
            rect = a.rect
            if content:
                print(f"--- Page {i+1} | Type: {subtype} | Author: {title} ---")
                print(f"Rect: {rect}")
                print(content)
                print()
            else:
                span = a.get_text().strip()
                if span:
                    print(f"--- Page {i+1} | Type: {subtype} (popup text) ---")
                    print(span)
                    print()

doc.close()