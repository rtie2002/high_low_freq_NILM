from pathlib import Path
import fitz

pdf_path = Path(r"C:\Users\Raymond Tie\Desktop\PhD\Code\multi-domain NILM\high_low_freq_NILM\_tai_hua_notice_qa\Tai_Hua_HR_Notice_Formatted.pdf")
out_dir = pdf_path.parent
pdf = fitz.open(pdf_path)
for index, page in enumerate(pdf):
    pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
    pix.save(out_dir / f"page-{index + 1}.png")
print(f"Rendered {len(pdf)} pages")
