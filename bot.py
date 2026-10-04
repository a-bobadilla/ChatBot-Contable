#!/usr/bin/env python3
"""
Bot de Telegram para Consultas y Asistente Contable Inteligente (RAG).
Proyecto Cátedra de Contabilidad - Facultad Politécnica - Universidad Nacional de Asunción (FP-UNA).

Conecta Telegram Bot con Ollama (LLM) y ChromaDB (Vector Store RAG).
"""

import asyncio
import html
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

# Telegram imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

# ChromaDB & LangChain imports
import chromadb
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Ollama client
import ollama

# ---------------------------------------------------------
# Configuración de Logging
# ---------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ContabilidadBot")

# ---------------------------------------------------------
# Carga de Variables de Entorno
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest").strip()
CHROMA_DIR = BASE_DIR / os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "contabilidad_erp").strip()
JSONL_CHUNKS_PATH = BASE_DIR / "processed_docs" / "chunks" / "knowledge_chunks.jsonl"
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "4"))

# Memoria conversacional básica en memoria (últimos N turnos por chat_id)
CONVERSATION_MEMORY: Dict[int, List[Dict[str, str]]] = {}
MAX_MEMORY_TURNS = 4  # Guarda últimos 4 pares de preguntas/respuestas


# ---------------------------------------------------------
# Clase de Embeddings Locales (ONNX all-MiniLM-L6-v2)
# ---------------------------------------------------------
class LocalChromaEmbeddings(Embeddings):
    """Embeddings locales ONNX independientes de APIs externas."""
    def __init__(self):
        self._fn = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._fn(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._fn([text])[0]


# ---------------------------------------------------------
# Gestor de Base de Conocimiento RAG (ChromaDB)
# ---------------------------------------------------------
class RAGKnowledgeBase:
    def __init__(self, chroma_dir: Path, collection_name: str, jsonl_path: Path):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.jsonl_path = jsonl_path
        self.embeddings = LocalChromaEmbeddings()
        self.vectorstore: Optional[Chroma] = None
        self.client: Optional[chromadb.ClientAPI] = None

    def initialize(self):
        """Inicializa ChromaDB e indexa los fragmentos si la colección está vacía."""
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Conectando con ChromaDB en: {self.chroma_dir}")

        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        collection = self.client.get_or_create_collection(self.collection_name)
        count = collection.count()
        logger.info(f"Colección '{self.collection_name}' contiene actualmente {count} fragmentos.")

        if count == 0 and self.jsonl_path.exists():
            logger.info(f"Colección vacía. Iniciando indexación automática desde: {self.jsonl_path}")
            self.index_chunks(reset=False)
        elif count == 0 and not self.jsonl_path.exists():
            logger.warning(f"No se encontró archivo de fragmentos en: {self.jsonl_path}. RAG operará sin documentos previos.")

        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )

    def index_chunks(self, reset: bool = False, batch_size: int = 250) -> int:
        """Lee el archivo JSONL e inserta los chunks en lotes en ChromaDB."""
        if not self.jsonl_path.exists():
            logger.error(f"Archivo JSONL no existe: {self.jsonl_path}")
            return 0

        if reset:
            logger.info("Reiniciando colección en ChromaDB...")
            try:
                self.client.delete_collection(self.collection_name)
            except Exception as e:
                logger.warning(f"Error al eliminar colección: {e}")
            self.client.get_or_create_collection(self.collection_name)

        documents = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    text = item.get("text", "").strip()
                    if not text:
                        continue
                    meta = item.get("metadata", {})
                    headings = meta.get("headings", [])
                    headings_str = " > ".join(headings) if isinstance(headings, list) else str(headings)

                    doc = Document(
                        page_content=text,
                        metadata={
                            "source_file": str(item.get("source_file", "desconocido")),
                            "headings": headings_str,
                            "chunk_id": f"chunk_{idx:05d}"
                        }
                    )
                    documents.append(doc)
                except Exception as ex:
                    logger.debug(f"Error parseando línea {idx}: {ex}")

        total_docs = len(documents)
        logger.info(f"Total documentos válidos a indexar: {total_docs}")

        temp_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )

        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            temp_store.add_documents(batch)
            logger.info(f"Progreso indexación: {min(i + batch_size, total_docs)}/{total_docs} fragmentos.")

        self.vectorstore = temp_store
        logger.info(f"Indexación finalizada exitosamente: {total_docs} fragmentos guardados.")
        return total_docs

    def search_similar(self, query: str, k: int = TOP_K_CHUNKS) -> List[Document]:
        """Realiza búsqueda de similitud semántica para la consulta."""
        if not self.vectorstore:
            return []
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error en búsqueda semántica: {e}")
            return []

    def get_document_count(self) -> int:
        try:
            if self.client:
                col = self.client.get_collection(self.collection_name)
                return col.count()
        except Exception:
            pass
        return 0


# ---------------------------------------------------------
# Gestor de LLM (Ollama)
# ---------------------------------------------------------
class OllamaManager:
    def __init__(self, host: str, model: str):
        self.host = host
        self.model = model
        self.client = ollama.Client(host=self.host)

    def check_health(self) -> Dict[str, Any]:
        """Verifica disponibilidad de Ollama y del modelo."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            if r.status_code == 200:
                data = r.json()
                models = [m.get("name") for m in data.get("models", [])]
                model_ready = any(self.model in m for m in models)
                return {
                    "online": True,
                    "models_available": models,
                    "target_model_ready": model_ready
                }
        except Exception as e:
            logger.warning(f"Fallo al conectar con Ollama en {self.host}: {e}")
        return {"online": False, "models_available": [], "target_model_ready": False}

    def generate_accounting_response(
        self,
        query: str,
        retrieved_docs: List[Document],
        chat_history: List[Dict[str, str]]
    ) -> str:
        """Construye el prompt con contexto RAG y genera respuesta con Ollama."""
        
        # Formatear contexto recuperado
        if retrieved_docs:
            context_blocks = []
            for i, doc in enumerate(retrieved_docs, start=1):
                src = doc.metadata.get("source_file", "Material de la cátedra")
                headings = doc.metadata.get("headings", "")
                header_info = f"[{i}] Fuente: {src}" + (f" | Sección: {headings}" if headings else "")
                context_blocks.append(f"{header_info}\n{doc.page_content}")
            context_str = "\n\n---\n\n".join(context_blocks)
        else:
            context_str = "No se recuperaron fragmentos específicos para esta consulta."

        system_prompt = (
            "Eres un Asistente y Tutor Contable Inteligente, diseñado para estudiantes de la Cátedra de "
            "Contabilidad de la Facultad Politécnica - Universidad Nacional de Asunción (FP-UNA).\n\n"
            "Tu objetivo es explicar de forma clara, didáctica, precisa y fundamentada conceptos y casos contables.\n\n"
            "DIRECTRICES OBLIGATORIAS:\n"
            "1. Utiliza primordialmente la información provista en el CONTEXTO DE LA BASE DE CONOCIMIENTO.\n"
            "2. CLASIFICACIÓN DE CUENTAS: Cuando el usuario pregunte por una cuenta (ej: Caja, Proveedores, Capital), "
            "indica con precisión su naturaleza (Activo, Pasivo, Patrimonio Neto, Ingresos, Costos o Gastos) "
            "y explica la razón conceptual (ej: representa bienes/derechos disponibles u obligaciones asumidas).\n"
            "3. ASIENTOS CONTABLES: Si solicitan un asiento contable o registrar una operación:\n"
            "   - Presenta las cuentas en una tabla clara con columnas: `Cuenta` | `Debe` | `Haber`.\n"
            "   - Aplica el principio de Partida Doble (Total Debe = Total Haber).\n"
            "   - Considera el IVA cuando corresponda (10% o 5% según la legislación paraguaya, distinguiendo IVA Crédito Fiscal e IVA Débito Fiscal).\n"
            "   - Añade una breve explicación de la variación patrimonial o justificación del asiento.\n"
            "4. TONO: Educativo, profesional, conciso y formal en español.\n"
            "5. Si el contexto no contiene suficiente información, respóndelo con bases contables sólidas universales "
            "o sugiere al alumno consultar el material específico de la cátedra.\n"
            "6. NO incluyas fuentes bibliográficas, citas, referencias de origen ni nombres de archivos al final ni en ninguna parte de tu respuesta (no agregues 'Fuente: ...'). Responde de forma directa, limpia y pedagógica.\n"
            "7. FORMATO DE NEGRITAS (TELEGRAM): Al destacar términos en negrita con asteriscos `**texto**`, asegúrate SIEMPRE de dejar un espacio antes del `**` de apertura y después del `**` de cierre (ejemplo: 'es una cuenta de **Activo** corriente', '• **Caja**: representa...'). NUNCA dejes espacios dentro de los asteriscos (usa `**texto**`, jamás `** texto **`)."
        )

        user_content = (
            f"=== CONTEXTO DE LA BASE DE CONOCIMIENTO ===\n"
            f"{context_str}\n\n"
            f"=== CONSULTA DEL ESTUDIANTE ===\n"
            f"{query}"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Incorporar memoria reciente
        for item in chat_history:
            messages.append({"role": item["role"], "content": item["content"]})

        messages.append({"role": "user", "content": user_content})

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.2,  # Baja temperatura para respuestas factuales y precisas
                    "top_p": 0.9,
                }
            )
            raw_content = response.get("message", {}).get("content", "No se obtuvo respuesta del modelo.")
            return clean_sources(raw_content)
        except Exception as e:
            logger.error(f"Error al generar respuesta en Ollama: {e}")
            return (
                f"⚠️ Ocurrió un error al consultar con el modelo `{self.model}` en Ollama:\n"
                f"`{str(e)}`\n\n"
                f"Por favor verifica que el servicio de Ollama esté ejecutándose correctamente."
            )


# ---------------------------------------------------------
# Tarea auxiliar para mantener acción "typing" en Telegram
# ---------------------------------------------------------
async def keep_typing(context: ContextTypes.DEFAULT_TYPE, chat_id: int, stop_event: asyncio.Event):
    """Envía la acción de 'escribiendo...' periódicamente mientras el LLM procesa."""
    while not stop_event.is_set():
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            pass


# ---------------------------------------------------------
# Utilidades de Formateo y Limpieza para Telegram
# ---------------------------------------------------------
def clean_sources(text: str) -> str:
    """Elimina cualquier sección final o mención de fuentes, bibliografía o nombres de archivo."""
    patterns = [
        # Sección final con Fuentes, Fuentes consultadas, Bibliografía, Referencias, etc.
        r'(?i)(?:\n+|\s{2,}|\s+)(?:\*+|\_{1,2})?\s*(?:fuentes?|bibliograf[íi]a|referencias?)[^\n:]*:\s*(?:\*+|\_{1,2})?.*$',
        r'(?i)\[\s*(?:fuente|referencia)[^\]]*\]',
        r'(?i)\(fuente:[^\)]*\)',
    ]
    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, '', cleaned, flags=re.DOTALL | re.MULTILINE)
    return cleaned.strip()


def format_telegram_bold(text: str) -> str:
    """
    Verifica y ajusta el formato de negritas con asteriscos (**texto**) para Telegram:
    1. Quita espacios internos accidentales dentro de los asteriscos (** texto ** -> **texto**).
    2. Asegura espacio antes de abrir ** si viene precedido de letras, números o signos como :, •, -, |, etc.
    3. Asegura espacio después de cerrar ** si viene seguido inmediatamente por una letra o número.
    """
    if not text:
        return text

    # 1. Limpiar espacios internos dentro de los asteriscos (** texto ** -> **texto**)
    text = re.sub(r'\*\*\s*([^\*\n]+?)\s*\*\*', lambda m: f"**{m.group(1).strip()}**" if m.group(1).strip() else "", text)

    # 2. Asegurar espacio antes de abrir ** si viene pegado a caracter que no sea espacio ni delimitador de apertura
    bold_pattern = re.compile(r'\*\*([^\*\s\n](?:[^\*\n]*?[^\*\s\n])?)\*\*')
    pieces = []
    last_end = 0

    for match in bold_pattern.finditer(text):
        start, end = match.span()
        bold_content = match.group(0)
        prefix = text[last_end:start]

        if prefix:
            last_char = prefix[-1]
            if last_char not in ' \t\r\n([{\\"\'«¡¿':
                prefix += ' '

        pieces.append(prefix)
        pieces.append(bold_content)
        last_end = end

    pieces.append(text[last_end:])
    rebuilt = ''.join(pieces)

    # 3. Asegurar espacio después de cerrar ** si está pegado a letra o número
    final_pattern = re.compile(r'(\*\*[^\*\s\n](?:[^\*\n]*?[^\*\s\n])?\*\*)([a-zA-Z0-9áéíóúÁÉÍÓÚñÑ])')
    rebuilt = final_pattern.sub(r'\1 \2', rebuilt)

    return rebuilt


def markdown_to_html(text: str) -> str:
    """Convierte Markdown básico (negritas, código e inline code) a HTML seguro para Telegram."""
    code_blocks = []
    def save_pre(m):
        code_blocks.append(m.group(2))
        return f"___CODE_BLOCK_{len(code_blocks)-1}___"
    t = re.sub(r'```([a-zA-Z0-9_-]*)\n?(.*?)```', save_pre, text, flags=re.DOTALL)

    inline_codes = []
    def save_inline(m):
        inline_codes.append(m.group(1))
        return f"___INLINE_CODE_{len(inline_codes)-1}___"
    t = re.sub(r'`([^`\n]+)`', save_inline, t)

    t = html.escape(t)
    t = re.sub(r'\*\*([^\*\n]+?)\*\*', r'<b>\1</b>', t)

    for i, c in enumerate(inline_codes):
        t = t.replace(f"___INLINE_CODE_{i}___", f"<code>{html.escape(c)}</code>")
    for i, b in enumerate(code_blocks):
        t = t.replace(f"___CODE_BLOCK_{i}___", f"<pre>{html.escape(b)}</pre>")

    return t


# ---------------------------------------------------------
# Utilidad para envío seguro de mensajes en Telegram
# ---------------------------------------------------------
async def send_smart_message(update: Update, text: str, reply_markup=None):
    """Envía mensajes dividiéndolos si superan el límite de 4096 caracteres y maneja formato seguro de Markdown y HTML."""
    processed_text = format_telegram_bold(clean_sources(text))
    MAX_LEN = 3900
    chunks = [processed_text[i:i + MAX_LEN] for i in range(0, len(processed_text), MAX_LEN)] if len(processed_text) > MAX_LEN else [processed_text]

    for idx, chunk in enumerate(chunks):
        current_markup = reply_markup if idx == len(chunks) - 1 else None
        
        # 1. Intentar enviar con Markdown
        try:
            await update.effective_message.reply_text(
                chunk,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=current_markup
            )
            continue
        except Exception as e:
            logger.warning(f"Error enviando mensaje con Markdown ({e}), intentando con HTML...")

        # 2. Fallback a HTML para preservar negritas y formato si Markdown falló
        try:
            html_chunk = markdown_to_html(chunk)
            await update.effective_message.reply_text(
                html_chunk,
                parse_mode=ParseMode.HTML,
                reply_markup=current_markup
            )
            continue
        except Exception as e_html:
            logger.warning(f"Error enviando mensaje con HTML ({e_html}), enviando en texto plano...")

        # 3. Fallback final en texto plano
        await update.effective_message.reply_text(
            chunk,
            parse_mode=None,
            reply_markup=current_markup
        )


# ---------------------------------------------------------
# Handlers de Telegram
# ---------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start: Bienvenida y menú rápido."""
    user = update.effective_user
    name = user.first_name if user else "Estudiante"

    welcome_text = (
        f"👋 ¡Hola, *{name}*!\n\n"
        f"Soy el **Tutor Contable Inteligente** de la Facultad Politécnica - Universidad Nacional de Asunción (FP-UNA).\n\n"
        f"Estoy entrenado con el material bibliográfico de la cátedra mediante **RAG** (ChromaDB) "
        f"y potenciado con **Ollama** (`{OLLAMA_MODEL}`).\n\n"
        f"📚 *¿En qué te puedo ayudar?*\n"
        f"• **Conceptos**: Activo, Pasivo, Patrimonio Neto, Debe, Haber, Partida Doble.\n"
        f"• **Clasificación de cuentas**: ¿Caja, Proveedores o Capital son Activo o Pasivo?\n"
        f"• **Asientos Contables**: Registro de compras, ventas, IVA (10% o 5%) y variaciones patrimoniales.\n"
        f"• **Libros Contables**: Libro Diario, Libro Mayor y Balance de Comprobación.\n\n"
        f"Puedes escribir tu pregunta directamente o presionar uno de los botones abajo:"
    )

    keyboard = [
        [
            InlineKeyboardButton("📖 ¿Qué es la Partida Doble?", callback_data="q_partida_doble"),
            InlineKeyboardButton("📊 ¿Caja es Activo o Pasivo?", callback_data="q_caja_activo"),
        ],
        [
            InlineKeyboardButton("📝 Ejemplo Asiento Compra con IVA", callback_data="q_asiento_compra"),
            InlineKeyboardButton("ℹ️ Estado del Sistema", callback_data="q_estado"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await send_smart_message(update, welcome_text, reply_markup=reply_markup)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /ayuda o /help: Explicación de comandos y ejemplos."""
    help_text = (
        "📖 **Guía de Uso del Chatbot Contable**\n\n"
        "Puedes realizar cualquier pregunta en lenguaje natural. Aquí tienes ejemplos prácticos:\n\n"
        "🔹 **Clasificación de Cuentas**:\n"
        "• *¿Proveedores qué tipo de cuenta es y cuál es su saldo habitual?*\n"
        "• *¿A qué grupo pertenece la cuenta Rodados?*\n\n"
        "🔹 **Conceptos y Principios**:\n"
        "• *¿Qué representa el Debe y el Haber en la contabilidad?*\n"
        "• *¿Cuál es la diferencia entre Libro Diario y Libro Mayor?*\n\n"
        "🔹 **Asientos Contables con IVA**:\n"
        "• *Realiza el asiento: Venta de mercaderías por Gs. 5.500.000 IVA incluido al contado.*\n"
        "• *Compra de mercaderías a crédito por Gs. 2.000.000 + IVA 10%.*\n\n"
        "🛠 **Comandos Disponibles**:\n"
        "• `/start` - Menú de inicio y presentación.\n"
        "• `/ayuda` - Esta guía de ayuda.\n"
        "• `/asiento <descripción>` - Solicitar directamente un asiento contable.\n"
        "• `/estado` - Ver estado de conexión con Ollama y ChromaDB.\n"
        "• `/limpiar` - Reiniciar el hilo de memoria de la conversación.\n"
        "• `/reindexar` - Reconstruir la base vectorial desde los documentos."
    )
    await send_smart_message(update, help_text)


async def cmd_asiento(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /asiento <operación>: Atajo para asientos contables."""
    if not context.args:
        await send_smart_message(
            update,
            "⚠️ Por favor indica la operación comercial después de `/asiento`.\n"
            "Ejemplo:\n`/asiento Compra de mercaderías por Gs. 2.200.000 IVA incluido en efectivo`"
        )
        return

    operation = " ".join(context.args)
    prompt = f"Realiza el asiento contable en formato tabla con cuentas, Debe y Haber para la siguiente operación: {operation}"
    await process_user_query(update, context, prompt)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /estado: Muestra diagnóstico de Ollama y ChromaDB."""
    rag_kb: RAGKnowledgeBase = context.bot_data["rag_kb"]
    ollama_mgr: OllamaManager = context.bot_data["ollama_mgr"]

    health = ollama_mgr.check_health()
    doc_count = rag_kb.get_document_count()

    ollama_status_icon = "🟢 Conectado" if health["online"] else "🔴 Desconectado"
    model_icon = "🟢 Disponible" if health["target_model_ready"] else "🟡 No encontrado en tags"

    chat_id = update.effective_chat.id
    history_len = len(CONVERSATION_MEMORY.get(chat_id, [])) // 2

    status_text = (
        "🖥 **Diagnóstico del Sistema Bot Contable**\n\n"
        f"• **Servicio Ollama**: {ollama_status_icon}\n"
        f"  - Host: `{OLLAMA_HOST}`\n"
        f"  - Modelo configurado: `{OLLAMA_MODEL}` ({model_icon})\n"
        f"  - Modelos disponibles en Ollama: `{', '.join(health['models_available']) if health['models_available'] else 'Ninguno'}`\n\n"
        f"• **Base Vectorial ChromaDB (RAG)**:\n"
        f"  - Colección: `{COLLECTION_NAME}`\n"
        f"  - Chunks indexados: *{doc_count:,} fragmentos*\n"
        f"  - Directorio: `{CHROMA_DIR}`\n\n"
        f"• **Memoria Conversacional Actual**:\n"
        f"  - Mensajes en memoria para este chat: *{history_len} turnos*"
    )
    await send_smart_message(update, status_text)


async def cmd_clear_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /limpiar: Borra la memoria conversacional del chat actual."""
    chat_id = update.effective_chat.id
    if chat_id in CONVERSATION_MEMORY:
        CONVERSATION_MEMORY[chat_id] = []
    await send_smart_message(
        update,
        "🧹 Memoria de la conversación restablecida. ¡Iniciamos un nuevo tema desde cero!"
    )


async def cmd_reindex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /reindexar: Reconstruye los embeddings en ChromaDB."""
    rag_kb: RAGKnowledgeBase = context.bot_data["rag_kb"]
    await send_smart_message(update, "⏳ Re-indexando fragmentos en ChromaDB, esto puede demorar un momento...")

    loop = asyncio.get_running_loop()
    try:
        count = await loop.run_in_executor(None, rag_kb.index_chunks, True, 250)
        await send_smart_message(update, f"✅ Re-indexación completada con éxito. Total: *{count:,}* fragmentos guardados.")
    except Exception as e:
        await send_smart_message(update, f"❌ Error durante la indexación: `{str(e)}`")


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja las pulsaciones de botones de la botonera inicial."""
    query = update.callback_query
    await query.answer()

    data = query.data
    queries_map = {
        "q_partida_doble": "¿Qué es la Partida Doble y cuáles son sus principios fundamentales en Contabilidad?",
        "q_caja_activo": "¿La cuenta Caja es Activo o Pasivo? Explica su clasificación y saldo.",
        "q_asiento_compra": "Realiza un asiento contable modelo de compra de mercaderías por Gs. 1.100.000 IVA 10% incluido, abonando 50% al contado en efectivo y 50% a crédito.",
    }

    if data == "q_estado":
        await cmd_status(update, context)
    elif data in queries_map:
        await process_user_query(update, context, queries_map[data])


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja mensajes de texto libres del usuario."""
    if not update.effective_message or not update.effective_message.text:
        return

    text = update.effective_message.text.strip()
    await process_user_query(update, context, text)


async def process_user_query(update: Update, context: ContextTypes.DEFAULT_TYPE, query_text: str):
    """Flujo RAG central: Búsqueda vectorial + Inferencia con Ollama."""
    chat_id = update.effective_chat.id
    rag_kb: RAGKnowledgeBase = context.bot_data["rag_kb"]
    ollama_mgr: OllamaManager = context.bot_data["ollama_mgr"]

    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(keep_typing(context, chat_id, stop_typing))

    loop = asyncio.get_running_loop()

    try:
        # 1. Búsqueda de similitud en ChromaDB (RAG)
        retrieved_docs = await loop.run_in_executor(
            None,
            rag_kb.search_similar,
            query_text,
            TOP_K_CHUNKS
        )

        # 2. Obtener historial del chat
        history = CONVERSATION_MEMORY.get(chat_id, [])

        # 3. Generar respuesta con Ollama
        response_text = await loop.run_in_executor(
            None,
            ollama_mgr.generate_accounting_response,
            query_text,
            retrieved_docs,
            history
        )

        # 4. Actualizar memoria de conversación
        history.append({"role": "user", "content": query_text})
        history.append({"role": "assistant", "content": response_text})
        if len(history) > MAX_MEMORY_TURNS * 2:
            history = history[-MAX_MEMORY_TURNS * 2:]
        CONVERSATION_MEMORY[chat_id] = history

    finally:
        stop_typing.set()
        await typing_task

    # 5. Enviar respuesta al usuario
    await send_smart_message(update, response_text)


# ---------------------------------------------------------
# Función Principal de Inicialización y Arranque
# ---------------------------------------------------------
def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("No se encontró TELEGRAM_BOT_TOKEN en las variables de entorno o archivo .env.")
        sys.exit(1)

    logger.info("=== Iniciando Bot de Contabilidad FP-UNA (RAG + Ollama) ===")
    logger.info(f"Ollama Host: {OLLAMA_HOST} | Modelo: {OLLAMA_MODEL}")
    logger.info(f"Ruta ChromaDB: {CHROMA_DIR}")

    # Inicializar Base de Conocimiento RAG
    rag_kb = RAGKnowledgeBase(
        chroma_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        jsonl_path=JSONL_CHUNKS_PATH
    )
    rag_kb.initialize()

    # Inicializar Gestor de Ollama
    ollama_mgr = OllamaManager(host=OLLAMA_HOST, model=OLLAMA_MODEL)
    health = ollama_mgr.check_health()
    if health["online"]:
        logger.info(f"Ollama responde correctamente. Modelos detectados: {health['models_available']}")
        if not health["target_model_ready"]:
            logger.warning(
                f"El modelo objetivo '{OLLAMA_MODEL}' aún no aparece en la lista de Ollama. "
                f"Asegúrate de descargarlo con: ollama pull {OLLAMA_MODEL}"
            )
    else:
        logger.warning(f"No fue posible comunicarse con Ollama en {OLLAMA_HOST}. El bot reintentará en cada consulta.")

    # Configurar Aplicación de Telegram
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Inyectar dependencias en bot_data para acceso en handlers
    app.bot_data["rag_kb"] = rag_kb
    app.bot_data["ollama_mgr"] = ollama_mgr

    # Registro de Handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler(["help", "ayuda"], cmd_help))
    app.add_handler(CommandHandler("asiento", cmd_asiento))
    app.add_handler(CommandHandler("estado", cmd_status))
    app.add_handler(CommandHandler("limpiar", cmd_clear_memory))
    app.add_handler(CommandHandler("reindexar", cmd_reindex))
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot de Telegram configurado con éxito. Iniciando sondeo (polling)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
