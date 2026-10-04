import argparse
import json
import sys
import time
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

DEFAULT_EXCLUDES = [
    "Indicadores de Evaluación_Proyecto Aúlico.pdf"
]

def setup_converter(enable_ocr: bool = True, accurate_tables: bool = True) -> DocumentConverter:
    """Configura el convertidor de Docling con PyPdfiumDocumentBackend (robusto ante rutas con tildes y caracteres especiales en Windows)."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    
    if accurate_tables:
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    else:
        pipeline_options.table_structure_options.mode = TableFormerMode.FAST
        
    pipeline_options.do_ocr = enable_ocr
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend
            )
        }
    )
    return converter

def process_pdf(
    pdf_path: Path, 
    md_output_dir: Path, 
    chunks_output_dir: Path = None,
    converter: DocumentConverter = None, 
    force: bool = False,
    with_chunks: bool = False
) -> bool:
    """Procesa un solo PDF y genera su Markdown y opcionalmente fragmentos semánticos en JSONL."""
    safe_name = pdf_path.stem
    output_md_path = md_output_dir / f"{safe_name}.md"
    
    if output_md_path.exists() and not force and not with_chunks:
        print(f"[SKIP] Ya procesado: {pdf_path.name} -> {output_md_path.name}")
        return True

    print(f"\n[INICIANDO] Procesando: {pdf_path.name} ({pdf_path.stat().st_size / (1024*1024):.2f} MB)...")
    start_time = time.time()
    
    try:
        result = converter.convert(pdf_path)
        markdown_text = result.document.export_to_markdown()
        
        md_output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
            
        num_tables = len(result.document.tables) if hasattr(result.document, "tables") else 0
        chunks_count = 0
        
        if with_chunks and chunks_output_dir:
            chunks_output_dir.mkdir(parents=True, exist_ok=True)
            chunks_file = chunks_output_dir / "knowledge_chunks.jsonl"
            
            try:
                from docling.chunking import HybridChunker
                chunker = HybridChunker()
                chunk_iter = chunker.chunk(result.document)
                
                with open(chunks_file, "a", encoding="utf-8") as jf:
                    for chunk in chunk_iter:
                        headings = []
                        if hasattr(chunk, "meta") and hasattr(chunk.meta, "headings") and chunk.meta.headings:
                            headings = list(chunk.meta.headings)
                        doc_items = 0
                        if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                            doc_items = len(chunk.meta.doc_items)
                            
                        chunk_record = {
                            "source_file": pdf_path.name,
                            "text": chunk.text,
                            "metadata": {
                                "headings": headings,
                                "doc_items": doc_items
                            }
                        }
                        jf.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
                        chunks_count += 1
            except Exception as chunk_err:
                print(f"[AVISO] No se pudieron extraer chunks para {pdf_path.name}: {chunk_err}", file=sys.stderr)

        elapsed = time.time() - start_time
        summary_info = f"[COMPLETADO] {pdf_path.name} en {elapsed:.2f}s | Tablas: {num_tables}"
        if with_chunks:
            summary_info += f" | Chunks semánticos: {chunks_count}"
        summary_info += f" -> {output_md_path.name}"
        print(summary_info)
        return True
    except Exception as e:
        print(f"[ERROR] Error procesando {pdf_path.name}: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Procesar documentos PDF con Docling")
    parser.add_argument("--docs-dir", type=str, default="docs", help="Directorio con los archivos PDF")
    parser.add_argument("--output-dir", type=str, default="processed_docs/markdown", help="Directorio para los archivos Markdown")
    parser.add_argument("--chunks-dir", type=str, default="processed_docs/chunks", help="Directorio para los chunks semánticos")
    parser.add_argument("--single-file", type=str, default=None, help="Nombre de un archivo específico para procesar")
    parser.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDES, help="Lista de nombres de archivos a excluir")
    parser.add_argument("--no-ocr", action="store_true", help="Desactivar OCR")
    parser.add_argument("--fast-tables", action="store_true", help="Usar modo de tablas rápido en lugar de preciso")
    parser.add_argument("--force", action="store_true", help="Forzar reprocesamiento si el archivo ya existe")
    parser.add_argument("--with-chunks", action="store_true", help="Generar también chunks semánticos con HybridChunker")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent.parent
    docs_path = (base_dir / args.docs_dir).resolve()
    output_path = (base_dir / args.output_dir).resolve()
    chunks_path = (base_dir / args.chunks_dir).resolve() if args.with_chunks else None
    
    if not docs_path.exists():
        print(f"Error: La carpeta de documentos '{docs_path}' no existe.", file=sys.stderr)
        sys.exit(1)
        
    output_path.mkdir(parents=True, exist_ok=True)
    
    exclude_set = set(args.exclude or [])
    
    if args.single_file:
        pdf_files = [docs_path / args.single_file]
        if not pdf_files[0].exists():
            print(f"Error: Archivo no encontrado: {pdf_files[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        all_pdfs = sorted(list(docs_path.glob("*.pdf")), key=lambda p: p.stat().st_size)
        pdf_files = [p for p in all_pdfs if p.name not in exclude_set]
        excluded_files = [p for p in all_pdfs if p.name in exclude_set]
        if excluded_files:
            print(f"Archivos excluidos por configuración ({len(excluded_files)}):")
            for ef in excluded_files:
                print(f"  - {ef.name}")

    print(f"\nTotal de archivos a procesar: {len(pdf_files)}")
    print(f"Directorio de origen: {docs_path}")
    print(f"Directorio Markdown: {output_path}")
    if args.with_chunks:
        print(f"Directorio Chunks: {chunks_path}")
    print("Inicializando modelos de Docling con PyPdfiumDocumentBackend...")
    
    converter = setup_converter(enable_ocr=not args.no_ocr, accurate_tables=not args.fast_tables)
    
    success_count = 0
    for pdf_file in pdf_files:
        if process_pdf(
            pdf_file, 
            output_path, 
            chunks_output_dir=chunks_path,
            converter=converter, 
            force=args.force,
            with_chunks=args.with_chunks
        ):
            success_count += 1
            
    print(f"\nResumen final: {success_count}/{len(pdf_files)} archivos procesados exitosamente.")

if __name__ == "__main__":
    main()
