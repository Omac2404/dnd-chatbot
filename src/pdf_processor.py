"""
PDF işleme modülü
"""

import pymupdf as fitz  # type: ignore
from pathlib import Path
from typing import List, Dict
from config import config


def extract_text_from_pdf(pdf_path: str) -> str:
    """PDF'den text çıkar"""
    doc = fitz.open(pdf_path)
    text = ""
    
    for page in doc:
        text += page.get_text()
    
    doc.close()
    return text


def get_pdf_metadata(pdf_path: str) -> Dict:
    """PDF metadata'sını al"""
    doc = fitz.open(pdf_path)
    
    metadata = {
        "filename": Path(pdf_path).name,
        "page_count": doc.page_count,
        "total_chars": sum(len(page.get_text()) for page in doc)
    }
    
    doc.close()
    return metadata


def list_pdfs() -> List[Path]:
    """data/pdfs/ klasöründeki PDF'leri listele"""
    if not config.PDF_DIR.exists():
        print(f"⚠️ {config.PDF_DIR} klasörü bulunamadı")
        return []
    
    pdfs = list(config.PDF_DIR.glob("*.pdf"))
    return pdfs


def main():
    """PDF processor test"""
    print("="*60)
    print("PDF PROCESSOR TEST")
    print("="*60 + "\n")
    
    # PDF'leri listele
    pdfs = list_pdfs()
    
    if not pdfs:
        print("❌ Hiç PDF bulunamadı!")
        print(f"PDF'lerinizi {config.PDF_DIR} klasörüne ekleyin")
        return
    
    print(f"✅ {len(pdfs)} PDF bulundu:\n")
    
    # Her PDF için metadata göster
    for pdf_path in pdfs:
        print(f"📄 {pdf_path.name}")
        meta = get_pdf_metadata(pdf_path)
        print(f"   Sayfa sayısı: {meta['page_count']}")
        print(f"   Karakter sayısı: {meta['total_chars']:,}")
        print()
    
    # İlk PDF'den örnek text çıkar
    if pdfs:
        print("="*60)
        print(f"Örnek Text Extraction: {pdfs[0].name}")
        print("="*60)
        
        text = extract_text_from_pdf(pdfs[0])
        print(f"İlk 500 karakter:\n{text[:500]}...")
        
        print("\n✅ PDF okuma başarılı!")


if __name__ == "__main__":
    main()