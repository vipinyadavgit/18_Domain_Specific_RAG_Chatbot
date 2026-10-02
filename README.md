# Automotive RAG Chatbot (Domain-Specific Document Q&A)

A beginner-friendly **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about **automotive software, standards and a vehicle user guide**, using only the documents in the `data/` folder.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Domain chosen](#2-domain-chosen)
3. [Dataset (documents used)](#3-dataset-documents-used)
4. [Architecture](#4-architecture)
5. [Project structure](#5-project-structure)
6. [Setup instructions](#6-setup-instructions)
7. [How to run ingestion](#7-how-to-run-ingestion)
8. [How to run the chatbot](#8-how-to-run-the-chatbot)
9. [Configuration (.env)](#9-configuration-env)
10. [Sample queries and outputs](#10-sample-queries-and-outputs)
11. [Bonus features included](#11-bonus-features-included)
12. [Limitations and assumptions](#12-limitations-and-assumptions)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Project overview

A normal LLM can "make up" answers. A RAG system avoids this:

1. Your documents are cut into small **chunks** and converted into **embeddings** (numbers that capture meaning).
2. The embeddings are saved in a **FAISS** vector database.
3. When you ask a question, the system finds the **most similar chunks**.
4. Only those chunks + your question are sent to the **LLM (OpenAI)**, which writes an answer **grounded in the documents** and adds **citations** like `[1]`.
5. If the answer is not in the documents, the bot says **"I don't know"**.

Two ways to chat: a **command-line chatbot** (required) and a **Streamlit web UI** (bonus).

## 2. Domain chosen

**Automotive software engineering** (the domain of my work at HARMAN).

The chatbot helps engineers and users with questions such as:

- Automotive standards: **AUTOSAR** (Classic and Adaptive), **ISO 26262** (functional safety, ASIL), **SOTIF (ISO 21448)**, **ISO/SAE 21434** and **UNECE R155/R156** (cybersecurity), **Automotive SPICE**
- In-vehicle networks and diagnostics: CAN, LIN, Ethernet, **UDS (ISO 14229)**
- Vehicle **user guide** topics: warning lights, infotainment, ADAS, maintenance, OTA updates, troubleshooting

## 3. Dataset (documents used)

The documents currently in `data/` are educational sample documents, not official copies of standards or a real vehicle manual.

| File | Type | Content |
|---|---|---|
| `automotive_software_standards_guide.pdf` | PDF | AUTOSAR, ISO 26262, SOTIF, ISO/SAE 21434, UNECE R155/R156, ASPICE, CAN/LIN/Ethernet, UDS, testing, SDV |
| `vehicle_user_guide.pdf` | PDF | Fictional "Sample Vehicle X1" owner's guide: warning lights, infotainment, ADAS, maintenance, tyres, troubleshooting, OTA |
| `automotive_glossary.txt` | TXT | Short glossary of automotive terms (ECU, ASIL, OTA, DTC, ...) |
| `infotainment_connectivity_faq.docx` | DOCX | FAQ about CarPlay/Android Auto, TCU/eSIM, infotainment OS, updates |

**Using your own documents:** just copy your real `.pdf`, `.txt` or `.docx` files into `data/` and run ingestion again. Multiple files are supported.

## 4. Architecture

```
                 INGESTION (run once, or whenever documents change)
 data/*.pdf,.txt,.docx
        |
        v
  1. Load documents  (pypdf / python-docx)      -> text + file name + page number
        |
        v
  2. Split into chunks (size + overlap)         -> small pieces of text
        |
        v
  3. Embeddings (OpenAI text-embedding-3-small) -> a vector for every chunk
        |
        v
  4. Save: FAISS index -> store/vector_index/   chunks (JSON) -> store/chunks/


                 CHAT (every question)
 User question
        |
        v
  5. Embed the question (same embedding model)
        |
        v
  6. FAISS similarity search -> top-k chunks (low-similarity chunks are dropped)
        |
        v
  7. Prompt = rules + numbered context + question  -> OpenAI chat model (gpt-4o-mini)
        |
        v
  8. Answer with citations [1], [2] + list of sources (file + page)
```

**Why these choices?**

- **FAISS `IndexFlatIP` + normalized vectors** = exact cosine-similarity search. Simple and accurate for small/medium document sets.
- **Chunk overlap** keeps sentences that fall on a chunk border readable.
- **`temperature=0`** makes the LLM stay factual.
- **MIN_SCORE filter**: if nothing is similar enough, the LLM is not even called and the bot answers "I don't know".

## 5. Project structure

```
18_Domain_Specific_RAG_Chatbot/
|
|-- app/
|   |-- config.py        # paths, settings from .env, logging, OpenAI client
|   |-- embeddings.py    # text -> vectors (OpenAI)
|   |-- ingest.py        # load -> chunk -> embed -> save to FAISS
|   |-- retriever.py     # question -> top-k relevant chunks
|   |-- rag_chain.py     # retrieved chunks + question -> LLM -> answer
|   |-- main.py          # command-line chatbot
|
|-- data/                # your documents (PDF / TXT / DOCX)
|-- store/
|   |-- vector_index/    # FAISS index (created by ingestion)
|   |-- chunks/          # chunk text + page info (created by ingestion)
|
|-- streamlit_app.py     # web UI (bonus)
|-- .env                 # YOUR secrets (never commit!)
|-- pyproject.toml       # dependencies for uv
|-- requirements.txt     # same dependencies for pip
|-- uv.lock              # locked dependency versions for uv
|-- README.md
```

## 6. Setup instructions

**Requirements:** Python 3.10 or newer, an OpenAI API key, internet access.

Run these steps in order from PowerShell. Replace the example path with the location of your project folder, the folder containing `pyproject.toml`.

### Step 1 - Go to the project folder

```powershell
cd "C:\path\to\18_Domain_Specific_RAG_Chatbot"
```

All commands below should be run from this project folder.

### Step 2 - Add your OpenAI API key

Open the existing `.env` file in the project folder and make sure `OPENAI_API_KEY` is set. If `.env` is missing in your copy, create it in the project folder and add that setting. Do not share the key or commit `.env` to GitHub; it is listed in `.gitignore`.

### Step 3 - Install the libraries (choose ONE option)

#### Option A - with `uv` (recommended)

If `uv` is not installed, install it once:

```powershell
python -m pip install uv
```

Then install the project libraries. This creates or updates `.venv` from the locked dependencies:

```powershell
uv sync
```

> If `uv` is not recognised after `pip install uv`, use `python -m uv sync` instead.

#### Option B - with plain `pip`

```powershell
# 1. Create a virtual environment inside the project folder
python -m venv .venv

# 2. Activate it (PowerShell). Your prompt will start with (.venv)
.\.venv\Scripts\Activate.ps1
#    If PowerShell blocks the script, run this once and try again:
#    Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# 3. Install all libraries
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check that the libraries are installed:

```powershell
python -c "import openai, faiss, numpy, pypdf, docx, dotenv, streamlit; print('All libraries OK')"
```

### Step 4 - Build the document index

The project already includes documents in `data/`. Ingestion reads them, creates embeddings, and saves the search index. Run it again whenever you add, remove, or change documents.

```powershell
# uv
uv run python -m app.ingest

# pip (with the .venv activated)
python -m app.ingest
```

### Step 5 - Start the chatbot

Choose either the command-line chatbot or the Streamlit web app:

```powershell
# Command line, with uv
uv run python -m app.main

# Or command line, with pip and the .venv activated
python -m app.main

# Or Streamlit web app, with uv
uv run streamlit run streamlit_app.py

# Or Streamlit web app, with pip and the .venv activated
streamlit run streamlit_app.py
```

### Libraries used

| Library | Why it is needed |
|---|---|
| `openai` | Embeddings and the chat model |
| `faiss-cpu` | Vector database / similarity search |
| `numpy` | Arrays for the vectors |
| `pypdf` | Read PDF files (gives page numbers) |
| `python-docx` | Read Word `.docx` files |
| `python-dotenv` | Read the `.env` file |
| `streamlit` | Web UI |

## 7. How to run ingestion

Ingestion reads the files in `data/`, creates chunks and embeddings and saves them into `store/`. Run it **once**, and again **whenever you add or change documents**.

```powershell
# with uv
uv run python -m app.ingest

# with pip (.venv activated)
python -m app.ingest
```

Expected output (numbers will differ with your files):

```
10:00:01 | INFO    | Loaded automotive_glossary.txt                       -> 1 page(s)/section(s)
10:00:01 | INFO    | Loaded automotive_software_standards_guide.pdf       -> 3 page(s)/section(s)
10:00:01 | INFO    | Loaded infotainment_connectivity_faq.docx            -> 1 page(s)/section(s)
10:00:01 | INFO    | Loaded vehicle_user_guide.pdf                        -> 2 page(s)/section(s)
10:00:01 | INFO    | Loaded 7 page(s)/section(s) in total
10:00:01 | INFO    | Created 30 chunks (size=800, overlap=100)
10:00:02 | INFO    | Creating embeddings with OpenAI (this can take a few seconds)...
10:00:04 | INFO    | Saved FAISS index to ...\store\vector_index\index.faiss
Done! 30 chunks are stored. You can now start the chatbot.
```

## 8. How to run the chatbot

### Command-line chatbot (required part)

```powershell
# with uv
uv run python -m app.main

# with pip (.venv activated)
python -m app.main
```

Type your question and press Enter. Type `exit` or `quit` to stop. Empty input and errors are handled.

### Streamlit web UI (bonus)

```powershell
# with uv
uv run streamlit run streamlit_app.py

# with pip (.venv activated)
streamlit run streamlit_app.py
```

Your browser opens at `http://localhost:8501`. The sidebar lets you change top-k, see the documents, re-build the index and clear the chat.

## 9. Configuration (.env)

All values except the API key are optional.

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | (required) | Your OpenAI key |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Model that writes the answer |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Model that creates embeddings |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Characters repeated between chunks |
| `TOP_K` | `4` | How many chunks are retrieved per question |
| `MIN_SCORE` | `0.10` | Minimum similarity (0 to 1) for a chunk to be used |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |

> After changing `CHUNK_SIZE`, `CHUNK_OVERLAP` or the embedding model, **run ingestion again**. The embedding model used for ingestion and for questions must be the same.

## 10. Sample queries and outputs

> These are **illustrative examples**. The exact wording from the LLM can differ slightly on your machine.

**Question:** `What are the ASIL levels in ISO 26262?`

```
Answer:
ISO 26262 defines four Automotive Safety Integrity Levels: ASIL A, B, C and D. ASIL A is the
lowest and ASIL D is the highest. Functions without safety relevance are rated QM (Quality
Management). The ASIL is found through the Hazard Analysis and Risk Assessment (HARA), which
rates Severity, Exposure and Controllability. [1]

Sources:
  [1] automotive_software_standards_guide.pdf, page 1  (similarity 0.71)
```

**Question:** `What does a red oil pressure warning light mean?`

```
Answer:
A red engine oil pressure light means the oil pressure is too low. Stop the engine
immediately and do not drive on. [1]

Sources:
  [1] vehicle_user_guide.pdf, page 1  (similarity 0.68)
```

**Question:** `How do I pair my phone with Bluetooth?`

```
Answer:
1. Turn Bluetooth on in your phone.
2. On the screen select Phone > Add Device.
3. Choose the vehicle name on your phone.
4. Confirm that the 6-digit code is the same on both devices. [1]
```

**Question:** `What is UDS service 0x27?`  ->  Security Access (from the UDS service list).

**Question:** `Who won the football world cup?` (not in the documents)

```
Answer:
I don't know. I could not find this in the automotive documents I have.
```

More questions to try:

- What is the difference between AUTOSAR Classic and Adaptive?
- What is ASIL decomposition?
- Which regulation requires a Software Update Management System?
- What is SOTIF?
- How often should engine oil be changed?
- What tyre pressure does Sample Vehicle X1 need?
- What are UDS negative responses?

## 11. Bonus features included

| Bonus from the assignment | Where |
|---|---|
| Streamlit web UI | `streamlit_app.py` |
| Source document + page number in answers | printed under every answer |
| Citations in answers | `[1]`, `[2]` inside the answer text |
| Support for `.txt` and `.docx` | `app/ingest.py` |
| Logging instead of print | `logging` in all modules |
| Configurable chunk size and top-k | `.env` (and a slider in the web UI) |

## 12. Limitations and assumptions

- The sample documents are a **short educational summary**, not the official text of AUTOSAR/ISO standards (those are copyrighted). Use your own approved documents for real work.
- The bot only knows what is in `data/`. If the answer is missing it says "I don't know".
- Scanned PDFs (images) have no text; `pypdf` cannot read them (OCR is not included).
- `.txt` and `.docx` files have no real pages, so no page number is shown for them.
- Each question is answered **independently**: the bot does not remember earlier questions in the same chat.
- PDF tables and images are not understood well, only extracted text.
- Retrieval uses vector search only (no keyword search), so exact IDs like `0x27` can sometimes rank lower than expected.
- Requires internet and an OpenAI account with credit; every ingestion and question costs a small amount of API usage.
- After adding or changing documents you must run ingestion again.

## 13. Troubleshooting

| Problem | Fix |
|---|---|
| `OPENAI_API_KEY is missing` | Put your key in `.env` after `OPENAI_API_KEY=` |
| `The vector store was not found` | Run ingestion first (section 7) |
| `No readable .pdf/.txt/.docx files found` | Put documents into the `data/` folder |
| `ModuleNotFoundError: No module named 'openai'` (or similar) | The libraries are not installed in the active environment. Redo Step 3 and make sure the `.venv` is activated (prompt starts with `(.venv)`) |
| `ModuleNotFoundError: No module named 'app'` | Run commands from the project root folder and use `python -m app.main` (not `python app/main.py`) |
| PowerShell says "running scripts is disabled" | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` and activate again |
| `uv` crashes or stops while downloading packages | Use **Option B (pip)** in Step 3 |
| `uv` is not recognised | Use `python -m uv ...` or install uv again |
| OpenAI error 401 / 429 | Wrong key, or no credit / rate limit. Check your OpenAI account |
