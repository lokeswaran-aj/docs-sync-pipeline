# Document Processing and Vector Storage Pipeline

A Python pipeline for processing documentation files, cleaning them, and storing them in a vector database for efficient retrieval and searching.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Features

- **Document Loading**: Supports MDX and MD files with flexible directory traversal
- **Content Cleaning**: 
  - Removes redundant whitespace from code blocks
  - Normalizes document formatting
  - Preserves code block metadata and language specifications
- **Smart Text Splitting**: Uses tiktoken-based splitting for optimal chunk sizes
- **Vector Storage**: Stores document embeddings in PostgreSQL for efficient similarity search
- **OpenAI Integration**: Leverages OpenAI's embedding models for high-quality vector representations

## 📋 Prerequisites

- Python 3.11 or higher
- PostgreSQL database
- OpenAI API key

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/lokeswaran-aj/docs-sync-pipeline.git
cd docs-sync-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` with your configuration:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/your_db
OPENAI_API_KEY=your_openai_api_key
EMBEDDING_MODEL=text-embedding-3-small
```

## 💻 Usage

1. Place your documentation files in the target directory:
```bash
mkdir -p repos/next.js/docs
# Add your .md or .mdx files to this directory
```

2. Run the processing pipeline:
```bash
python src/main.py
```

## 🏗️ Architecture

The pipeline consists of several key components:

1. **Document Loading**: Uses `DirectoryLoader` to recursively load markdown files
2. **Content Processing**: 
   - Cleans and normalizes document content
   - Preserves important formatting and metadata
3. **Text Splitting**: 
   - Chunks documents into optimal sizes (1000 tokens)
   - Maintains context with overlapping chunks (200 tokens)
4. **Vector Storage**:
   - Generates embeddings using OpenAI's models
   - Stores vectors in PostgreSQL using pgvector

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for document processing tools
- [pgvector](https://github.com/pgvector/pgvector) for vector similarity search in PostgreSQL
