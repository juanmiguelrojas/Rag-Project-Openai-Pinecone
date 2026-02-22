# RAG Project – OpenAI + Pinecone

> **Lab:** Introducción a la Creación de RAGs (Generadores con Recuperación Aumentada) con OpenAI  
> **Repositorio 2** – Implementación completa usando LangChain v0.2+, OpenAI y Pinecone.

---

## Tabla de Contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Arquitectura](#arquitectura)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Requisitos Previos](#requisitos-previos)
5. [Configuración e Instalación](#configuración-e-instalación)
6. [Variables de Entorno](#variables-de-entorno)
7. [Ejecución del Notebook](#ejecución-del-notebook)
8. [Salidas de Ejemplo](#salidas-de-ejemplo)
9. [Componentes Clave](#componentes-clave)
10. [Cómo Ejecutar el Notebook](#cómo-ejecutar-el-notebook)
11. [Cómo Guardar el Notebook con Outputs](#cómo-guardar-el-notebook-con-outputs)
12. [Referencias](#referencias)

---

## Descripción del Proyecto

Este repositorio implementa un pipeline de **Generación con Recuperación Aumentada (RAG)** que:

1. Carga documentos de texto plano (`.txt`) desde un directorio local `data/`.
2. Los divide en fragmentos con solapamiento.
3. Vectoriza los fragmentos con **OpenAI `text-embedding-3-small`**.
4. Almacena y recupera vectores en **Pinecone** (índice serverless o basado en pods).
5. Responde preguntas en lenguaje natural recuperando los fragmentos más relevantes y pasándolos como contexto a **OpenAI `gpt-4o-mini`** a través de una cadena **LangChain LCEL**.

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│                                                                  │
│  data/*.txt  ──►  DirectoryLoader/TextLoader                    │
│                         │                                        │
│                         ▼                                        │
│              RecursiveCharacterTextSplitter                      │
│           (chunk_size=1000, chunk_overlap=200)                   │
│                         │                                        │
│                         ▼                                        │
│             OpenAIEmbeddings (text-embedding-3-small)            │
│                         │                                        │
│                         ▼                                        │
│              PineconeVectorStore.from_documents()                │
│                    (upsert vectors)                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                           │
│                                                                  │
│  user question                                                   │
│       │                                                          │
│       ├──► VectorStoreRetriever ──► top-k chunks from Pinecone  │
│       │                                    │                     │
│       │                             format_docs()               │
│       │                                    │                     │
│       └───────────────────────────► ChatPromptTemplate          │
│                                            │                     │
│                                   ChatOpenAI (gpt-4o-mini)      │
│                                            │                     │
│                                     StrOutputParser             │
│                                            │                     │
│                                      final answer               │
└─────────────────────────────────────────────────────────────────┘
```

**Resumen de componentes:**

| Componente | Tecnología | Propósito |
|-----------|-----------|---------|
| Cargador de documentos | LangChain `DirectoryLoader` + `TextLoader` | Cargar archivos `.txt` |
| Divisor de texto | `RecursiveCharacterTextSplitter` | Dividir documentos en fragmentos |
| Embeddings | OpenAI `text-embedding-3-small` | Vectorizar fragmentos |
| Almacén de vectores | Pinecone (via `langchain-pinecone`) | Almacenar y recuperar vectores |
| Recuperador | `VectorStoreRetriever` | Búsqueda por similitud (top-k) |
| LLM | OpenAI `gpt-4o-mini` | Generar respuestas |
| Cadena | LangChain LCEL | Conectar recuperador → prompt → LLM |

---

## Estructura del Proyecto

```
Rag-Project-Openai-Pinecone/
├── .env.example                          # Plantilla de variables de entorno
├── requirements.txt                      # Dependencias de Python
├── README.md                             # Este archivo
│
├── data/
│   ├── README.md                         # Instrucciones para agregar archivos TXT
│   ├── ai_overview.txt                   # Documento de ejemplo – Visión general de IA
│   ├── ml_basics.txt                     # Documento de ejemplo – Fundamentos de ML
│   └── llm_intro.txt                     # Documento de ejemplo – Introducción a LLM
│
├── notebooks/
│   └── 01_rag_openai_pinecone_txt.ipynb  # Notebook principal de Jupyter
│
└── src/
    ├── __init__.py
    └── rag_utils.py                      # Funciones auxiliares reutilizables del pipeline
```

---

## Requisitos Previos

- Python 3.9+
- Una cuenta de **OpenAI** con una clave API – [https://platform.openai.com](https://platform.openai.com)
- Una cuenta de **Pinecone** con una clave API – [https://app.pinecone.io](https://app.pinecone.io)
- Un índice de Pinecone creado con **dimension = 1536** y **metric = cosine**

---

## Configuración e Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/juanmiguelrojas/Rag-Project-Openai-Pinecone.git
cd Rag-Project-Openai-Pinecone
```

### 2. Crear y activar un entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar las variables de entorno

```bash
cp .env.example .env
# Edita .env y completa con tus claves API
```

### 5. Crear el índice en Pinecone

Puedes hacerlo desde la [Consola de Pinecone](https://app.pinecone.io) o usando el **Paso 4** del notebook.

| Parámetro | Valor |
|---------|-------|
| Nombre | valor de `PINECONE_INDEX_NAME` en tu `.env` |
| Dimensiones | `1536` |
| Métrica | `cosine` |

### 6. Lanzar Jupyter y abrir el notebook

```bash
jupyter notebook notebooks/01_rag_openai_pinecone_txt.ipynb
```

Ejecuta todas las celdas de arriba hacia abajo.

---

## Variables de Entorno

Copia `.env.example` a `.env` y proporciona los siguientes valores:

| Variable | Requerida | Descripción |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | Clave API de OpenAI |
| `PINECONE_API_KEY` | ✅ | Clave API de Pinecone |
| `PINECONE_INDEX_NAME` | ✅ | Nombre de tu índice en Pinecone |
| `PINECONE_NAMESPACE` | ❌ | Namespace dentro del índice (valor por defecto: `default`) |

`.env.example`:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=rag-openai-index
PINECONE_NAMESPACE=default
```

---

## Ejecución del Notebook

El notebook (`notebooks/01_rag_openai_pinecone_txt.ipynb`) está dividido en 9 pasos:

| Paso | Descripción |
|------|-------------|
| 1 | Cargar las variables de entorno desde `.env` |
| 2 | Cargar los documentos TXT desde `data/` |
| 3 | Dividir los documentos en fragmentos con solapamiento |
| 4 | Verificar / crear el índice en Pinecone |
| 5 | Vectorizar los fragmentos y subir los vectores a Pinecone |
| 6 | *(Opcional)* Conectar a un vectorstore existente (omite la re-ingesta) |
| 7 | Construir la cadena LCEL RAG |
| 8 | Ejecutar consultas RAG de ejemplo |
| 9 | Inspeccionar los documentos fuente recuperados |

---

## Salidas de Ejemplo

### Carga de documentos (Paso 2)

```
100%|██████████| 3/3 [00:00<00:00, 45.12it/s]
Loaded 3 document(s) from '../data'.

--- First document preview ---
Source : ../data/ai_overview.txt
Length : 1560 characters

First 300 characters:
Artificial Intelligence (AI) is the simulation of human intelligence processes by machines...
```

### División en fragmentos (Paso 3)

```
Split into 8 chunk(s) (chunk_size=1000, overlap=200).

Total chunks : 8

--- Sample chunk ---
Artificial Intelligence (AI) is the simulation of human intelligence processes by machines,
especially computer systems. These processes include learning ...

Metadata: {'source': '../data/ai_overview.txt'}
```

### Subida a Pinecone (Paso 5)

```
Upserted 8 chunk(s) to Pinecone index 'rag-openai-index' (namespace='default').

✅  Vectors upserted successfully.
```

### Consulta RAG (Paso 8)

```
❓ Question: What is Retrieval-Augmented Generation (RAG) and why is it useful?

💬 Answer:
Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs)
by coupling them with a retrieval system. Instead of relying solely on parametric knowledge
(the model's weights), the model retrieves relevant documents from an external knowledge base
at inference time, grounding its responses in factual, up-to-date information. RAG reduces
hallucinations and allows the model to cite its sources.
```

### Inspección de fuentes (Paso 9)

```
🔍  Retrieved 4 chunk(s) for: "What are the limitations of Large Language Models?"

── Chunk 1 ── source: ../data/llm_intro.txt
Large Language Models (LLMs) are a type of artificial intelligence model trained on vast
amounts of text data to understand and generate human-like language ...

── Chunk 2 ── source: ../data/llm_intro.txt
Limitations of LLMs include hallucinations (generating plausible-sounding but incorrect
information), knowledge cutoffs, high computational cost ...

📚  Sources used:
 - ../data/llm_intro.txt
 - ../data/llm_intro.txt
 - ../data/ai_overview.txt
 - ../data/ml_basics.txt
```

---

## Componentes Clave

### `src/rag_utils.py`

Funciones auxiliares reutilizables para mantener el notebook legible:

| Función | Descripción |
|----------|-------------|
| `load_txt_documents(data_dir)` | Carga todos los archivos `.txt` de un directorio |
| `split_documents(docs, chunk_size, chunk_overlap)` | Divide los documentos en fragmentos con solapamiento |
| `build_vectorstore(chunks, index_name, namespace, embedding_model)` | Vectoriza y sube los fragmentos a Pinecone |
| `load_vectorstore(index_name, namespace, embedding_model)` | Conecta a un índice de Pinecone existente |
| `build_rag_chain(vectorstore, top_k, llm_model, temperature)` | Construye la cadena LCEL RAG |

---

## Cómo Ejecutar el Notebook

Sigue estos pasos para configurar el entorno y ejecutar el notebook desde cero:

### 1. Crear y activar un entorno virtual de Python

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 2. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 3. Crear el archivo `.env` a partir de la plantilla

```bash
cp .env.example .env
```

Abre `.env` en tu editor de texto y rellena cada variable con tus credenciales reales:

```dotenv
OPENAI_API_KEY=sk-...             # Tu clave API de OpenAI
PINECONE_API_KEY=...              # Tu clave API de Pinecone
PINECONE_INDEX_NAME=rag-openai-index
PINECONE_NAMESPACE=default
```

### 4. Lanzar Jupyter Notebook

```bash
jupyter notebook notebooks/01_rag_openai_pinecone_txt.ipynb
```

Esto abrirá el notebook `notebooks/01_rag_openai_pinecone_txt.ipynb` directamente en tu navegador.

### 5. Ejecutar las celdas

Una vez abierto el notebook en Jupyter, ejecuta las celdas en orden de arriba hacia abajo.  
Puedes hacerlo usando **Shift + Enter** celda por celda, o seleccionando **Kernel → Restart & Run All** para ejecutarlas todas de una vez.

---

## Cómo Guardar el Notebook con Outputs

Para conservar los resultados de ejecución (outputs) dentro del propio archivo del notebook:

### 1. Ejecutar todas las celdas

Desde el menú de Jupyter, selecciona:

```
Kernel → Restart & Run All
```

Espera a que todas las celdas terminen de ejecutarse (verás los outputs debajo de cada celda).

### 2. Guardar el notebook con outputs

Una vez finalizada la ejecución, guarda el archivo con:

- **Ctrl + S** (Windows/Linux) o **Cmd + S** (macOS), o bien
- **File → Save and Checkpoint**

El archivo `.ipynb` almacenará los outputs directamente en su estructura JSON.

### 3. Confirmar el notebook en el repositorio (opcional)

Si deseas incluir los outputs en el control de versiones, haz commit del archivo:

```bash
git add notebooks/01_rag_openai_pinecone_txt.ipynb
git commit -m "Add notebook with execution outputs"
git push
```

> ⚠️ **Importante:** Antes de hacer commit, revisa los outputs del notebook para asegurarte de que **no contienen claves API, tokens ni ningún tipo de secreto**. Las variables de entorno se cargan desde `.env` y no deberían aparecer en los outputs, pero verifica que ninguna celda imprima accidentalmente el valor de una credencial.

---

## Referencias

- [LangChain LLM Chain Quickstart](https://python.langchain.com/docs/tutorials/llm_chain/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LangChain Pinecone Integration](https://python.langchain.com/docs/integrations/vectorstores/pinecone/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
