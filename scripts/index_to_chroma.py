import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List
from tqdm import tqdm

from langchain_chroma import Chroma
from langchain_core.documents import Document
from chromadb.utils import embedding_functions

class LocalChromaEmbeddings:
    """Implementa la interfaz de LangChain Embeddings usando el modelo local ONNX all-MiniLM-L6-v2."""
    def __init__(self):
        self._fn = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._fn(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._fn([text])[0]

def get_embeddings(use_ollama: bool = False, ollama_url: str = "http://localhost:11434", ollama_model: str = "nomic-embed-text"):
    if use_ollama:
        from langchain_ollama import OllamaEmbeddings
        print(f"Utilizando OllamaEmbeddings ({ollama_model}) en {ollama_url}...")
        return OllamaEmbeddings(base_url=ollama_url, model=ollama_model)
    else:
        print("Utilizando Embeddings locales (ONNX all-MiniLM-L6-v2, 100% integrado y estable)...")
        return LocalChromaEmbeddings()

def index_chunks(
    jsonl_path: Path,
    chroma_dir: Path,
    collection_name: str = "contabilidad_erp",
    batch_size: int = 250,
    reset: bool = False,
    use_ollama: bool = False,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "nomic-embed-text"
):
    if not jsonl_path.exists():
        print(f"[ERROR] No se encontró el archivo de chunks: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    if reset and chroma_dir.exists():
        print(f"[INFO] Limpiando base de datos previa en: {chroma_dir}")
        shutil.rmtree(chroma_dir, ignore_errors=True)

    chroma_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings(use_ollama, ollama_url, ollama_model)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    print(f"\nCargando fragmentos desde: {jsonl_path}")
    raw_chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_chunks.append(json.loads(line))

    total_chunks = len(raw_chunks)
    print(f"Total de fragmentos en el archivo: {total_chunks}")

    documents = []
    for idx, item in enumerate(raw_chunks):
        text = item.get("text", "").strip()
        if not text:
            continue
            
        raw_meta = item.get("metadata", {})
        headings_list = raw_meta.get("headings", [])
        headings_str = " > ".join(headings_list) if isinstance(headings_list, list) else str(headings_list)
        
        doc = Document(
            page_content=text,
            metadata={
                "source_file": str(item.get("source_file", "unknown")),
                "headings": headings_str,
                "doc_items": int(raw_meta.get("doc_items", 0)),
                "chunk_id": f"chunk_{idx:05d}"
            }
        )
        documents.append(doc)

    print(f"Documentos válidos a indexar: {len(documents)}")
    start_time = time.time()

    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings
    )

    for i in tqdm(range(0, len(documents), batch_size), desc="Indexando en ChromaDB"):
        batch = documents[i:i + batch_size]
        vectorstore.add_documents(batch)

    elapsed = time.time() - start_time
    print(f"\n[ÉXITO] Indexación completada en {elapsed:.2f}s.")
    print(f"Colección: '{collection_name}' | Total chunks indexados: {len(documents)}")
    print(f"Base de datos persistida en: {chroma_dir.resolve()}\n")

def test_query(chroma_dir: Path, collection_name: str, query: str, n_results: int = 3):
    """Ejecuta una consulta de prueba sobre la colección indexada mediante LangChain."""
    print(f"=== Prueba de Búsqueda Semántica con LangChain + ChromaDB ===")
    print(f"Pregunta: '{query}'\n")
    
    embeddings = LocalChromaEmbeddings()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": n_results})
    results = retriever.invoke(query)

    for rank, doc in enumerate(results, start=1):
        meta = doc.metadata
        print(f"[{rank}] Fuente: {meta.get('source_file')}")
        if meta.get("headings"):
            print(f"    Sección: {meta.get('headings')}")
        snippet = doc.page_content[:260].replace("\n", " ") + "..." if len(doc.page_content) > 260 else doc.page_content.replace("\n", " ")
        print(f"    Extracto: {snippet}\n")

def main():
    parser = argparse.ArgumentParser(description="Indexar chunks de Docling en ChromaDB usando LangChain")
    parser.add_argument("--jsonl", type=str, default="processed_docs/chunks/knowledge_chunks.jsonl", help="Ruta a knowledge_chunks.jsonl")
    parser.add_argument("--chroma-dir", type=str, default="chroma_db", help="Directorio de persistencia de ChromaDB")
    parser.add_argument("--collection", type=str, default="contabilidad_erp", help="Nombre de la colección")
    parser.add_argument("--batch-size", type=int, default=250, help="Tamaño de lote para inserción")
    parser.add_argument("--reset", action="store_true", help="Reiniciar la colección antes de indexar")
    parser.add_argument("--use-ollama", action="store_true", help="Usar embeddings de Ollama")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="URL de Ollama")
    parser.add_argument("--ollama-model", type=str, default="nomic-embed-text", help="Modelo de embeddings en Ollama")
    parser.add_argument("--test-query", type=str, default=None, help="Ejecutar una consulta semántica de prueba")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent.parent
    jsonl_path = (base_dir / args.jsonl).resolve()
    chroma_dir = (base_dir / args.chroma_dir).resolve()
    
    if args.test_query:
        test_query(chroma_dir, args.collection, args.test_query)
    else:
        index_chunks(
            jsonl_path=jsonl_path,
            chroma_dir=chroma_dir,
            collection_name=args.collection,
            batch_size=args.batch_size,
            reset=args.reset,
            use_ollama=args.use_ollama,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model
        )
        test_query(chroma_dir, args.collection, "Qué es el Debe y el Haber en la partida doble?", n_results=3)

if __name__ == "__main__":
    main()
