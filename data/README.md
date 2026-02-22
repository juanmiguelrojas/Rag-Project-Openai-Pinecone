# Data Directory

Place your TXT documents here for ingestion into the RAG pipeline.

## Usage

1. Add one or more `.txt` files to this directory.
2. The notebook (`notebooks/01_rag_openai_pinecone_txt.ipynb`) will automatically discover and load all `.txt` files in this directory.
3. Each file will be split into overlapping chunks, embedded with OpenAI, and upserted into Pinecone.

## Sample Files

Three sample documents are included to demonstrate the pipeline:

- `ai_overview.txt` – A brief overview of Artificial Intelligence.
- `ml_basics.txt` – An introduction to Machine Learning concepts.
- `llm_intro.txt` – An introduction to Large Language Models.

## Notes

- Files must be **plain text** (UTF-8 encoding recommended).
- File names become part of the document metadata (`source` field), so use descriptive names.
- There is no hard limit on file size, but very large files may slow down embedding. Consider splitting them manually if needed.
