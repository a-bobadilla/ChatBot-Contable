# Intelligent Accounting Tutor Bot (RAG + Ollama + Telegram)

An intelligent Telegram tutor bot powered by Retrieval-Augmented Generation (RAG) and local LLM inference. Designed to assist accounting students by answering theoretical and practical queries, explaining accounting entries, and classifying accounts using course materials stored in a local ChromaDB vector database.

---

## Architecture & Features

- **Telegram Interface**: Built with `python-telegram-bot` with conversational memory and formatted Markdown answers.
- **Retrieval-Augmented Generation (RAG)**: Uses **ChromaDB** with built-in ONNX embeddings (`all-MiniLM-L6-v2`) to retrieve relevant sections from accounting course documents.
- **Local LLM Inference**: Communicates with a host-managed **Ollama** instance (`llama3.1`) to ensure full privacy, local execution, and optional GPU hardware acceleration.
- **Dockerized**: Containerized deployment with Docker and Docker Compose.

---

## Prerequisites

Before getting started, make sure you have installed on your host machine:

- [Git](https://git-scm.com/)
- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/) (v2.0+)
- [Ollama](https://ollama.com/) (running natively on the host machine)
- A Telegram Bot Token (obtained from [@BotFather](https://t.me/BotFather))

---

## Step-by-Step Setup Guide

### Step 1: Clone the Repository

Clone the project repository to your local machine:

```bash
# Using SSH
git clone git@github.com:a-bobadilla/Contabilida_ERP.git

# Or using HTTPS
git clone https://github.com/a-bobadilla/Contabilida_ERP.git

# Enter the project folder
cd Contabilida_ERP
```

*(If your directory was cloned or renamed to another folder name such as `Bot_Josh`, navigate into that folder).*

---

### Step 2: Install and Configure Ollama Locally

Running Ollama on the host machine enables direct access to host hardware acceleration (NVIDIA CUDA, AMD ROCm/Vulkan, or Apple Silicon).

#### 1. Install Ollama

- **Linux**:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- **macOS / Windows**:
  Download the installer from [ollama.com/download](https://ollama.com/download).

#### 2. Configure Host Network Binding (Crucial for Docker)

By default, Ollama only listens on `127.0.0.1:11434`. To allow the Docker container to communicate with Ollama via `host.docker.internal`, Ollama must bind to `0.0.0.0`:

- **Option A: Linux systemd service (Recommended)**
  Edit the systemd service or create a drop-in override:
  ```bash
  sudo systemctl edit ollama.service
  ```
  Add the following lines under `[Service]`:
  ```ini
  [Service]
  Environment="OLLAMA_HOST=0.0.0.0:11434"
  Environment="OLLAMA_KEEP_ALIVE=24h"
  ```
  Then reload and restart the service:
  ```bash
  sudo systemctl daemon-reload
  sudo systemctl restart ollama
  ```

- **Option B: Manual execution / Terminal**
  ```bash
  OLLAMA_HOST=0.0.0.0:11434 ollama serve
  ```

#### 3. Pull the Language Model

Download the default LLM model (`llama3.1`):

```bash
ollama pull llama3.1
```

Verify that the model is installed and Ollama is responding:

```bash
curl http://localhost:11434
# Should return: "Ollama is running"

ollama list
# Should display llama3.1 in the list
```

---

### Step 3: Configure Environment Variables

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Open `.env` with your preferred text editor (e.g. `nano .env`) and set your credentials:

   ```ini
   # Telegram Bot Token from @BotFather (Required)
   TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ

   # Ollama Host URL (Optional, defaults to http://host.docker.internal:11434)
   OLLAMA_HOST=http://host.docker.internal:11434

   # Ollama Model Name (Optional, defaults to llama3.1:latest)
   OLLAMA_MODEL=llama3.1:latest
   ```

---

### Step 4: Build and Deploy with Docker

Start the bot container using Docker Compose:

```bash
docker compose up -d --build
```

#### Verify Deployment

1. Check that the container is up and running:
   ```bash
   docker compose ps
   ```

2. Inspect the live logs to verify connection with Telegram and Ollama:
   ```bash
   docker compose logs -f bot
   ```

   You should see output similar to:
   ```text
   telegram_bot | ContabilidadBot - === Iniciando Bot de Contabilidad FP-UNA (RAG + Ollama) ===
   telegram_bot | ContabilidadBot - Verificando conexión con Ollama en http://host.docker.internal:11434...
   telegram_bot | ContabilidadBot - Conexión con Ollama exitosa. Modelo llama3.1:latest disponible.
   telegram_bot | telegram.ext.Application - Application started
   ```

---

## Bot Commands in Telegram

Once the bot is online, start a conversation with it on Telegram using the following commands:

| Command | Description |
| :--- | :--- |
| `/start` | Initializes the session, introduces the tutor, and explains how to ask questions. |
| `/help` or `/ayuda` | Shows tips on prompting, supported queries, and bot capabilities. |
| `/asiento` | Explains the structure and debit/credit logic of accounting entries. |
| `/estado` | Displays diagnostic information: Ollama connectivity, active model, and total documents indexed in ChromaDB. |
| `/limpiar` | Resets the short-term conversational history for your current chat session. |
| `/reindexar` | Rebuilds the ChromaDB vector database index from course documents. |

---

## Managing the Service

- **View Live Logs**:
  ```bash
  docker compose logs -f bot
  ```

- **Restart the Bot**:
  ```bash
  docker compose restart bot
  ```

- **Stop the Container**:
  ```bash
  docker compose down
  ```

- **Update Code and Rebuild**:
  ```bash
  git pull origin main
  docker compose up -d --build
  ```

---

## Re-indexing Knowledge Base (Optional)

The bot comes with pre-indexed knowledge under `chroma_db/`. If you add new documents to `docs/` or `processed_docs/` and wish to rebuild the vector database manually, you can execute the indexing script:

```bash
docker compose exec bot python scripts/index_to_chroma.py
```
Or directly send the `/reindexar` command to the bot on Telegram.

---

## Troubleshooting

### Container cannot connect to Ollama (`Connection Refused`)
1. Ensure Ollama is listening on `0.0.0.0:11434` and not strictly `127.0.0.1`:
   ```bash
   ss -tulpn | grep 11434
   ```
2. If using UFW or a firewall on Linux, allow connections from the Docker network:
   ```bash
   sudo ufw allow in on docker0 to any port 11434
   ```
   Or allow the local subnet (`172.17.0.0/16`).
3. Verify that `extra_hosts` in [docker-compose.yml](file:///home/s1laar/Bot_Josh/docker-compose.yml) includes:
   ```yaml
   extra_hosts:
     - "host.docker.internal:host-gateway"
   ```

### Invalid Telegram Token
Ensure that there are no quotes or extra spaces around the token in your `.env` file:
```ini
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
```
Restart the container with `docker compose restart bot`.

---

## ☕ Knowledge Base & Support

The bot relies on a comprehensive accounting knowledge base (syllabus, account catalogs, theoretical notes, and practical exam exercises) to power its RAG retrieval.

If you would like to obtain the curated knowledge base ready to use with the bot, or if you would like to support the ongoing development of this project, consider buying me a coffee:

👉 **[Buy me a coffee - Ariel Bobadilla](https://buymeacoffee.com/coco.xor)**

*(Once you've made a contribution, feel free to reach out to receive the full knowledge base dataset ready to drop into your `chroma_db/` folder).*

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
