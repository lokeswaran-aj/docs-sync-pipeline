import os
import re

from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_code_blocks(text: str) -> str:
    """Remove extra whitespace from code blocks
    while preserving regular text."""
    # Pattern to match code blocks with all possible metadata
    pattern = r"```(?:.*?)\n([\s\S]*?)```"

    def clean_block(match):
        code = match.group(1)
        # Remove empty lines and normalize whitespace in code
        cleaned = "\n".join(line for line in code.split("\n") if line.strip())
        return f"```\n{cleaned}\n```"

    # Replace code blocks with cleaned versions
    cleaned_text = re.sub(pattern, clean_block, text, flags=re.DOTALL)

    # Remove multiple consecutive newlines from the entire document
    cleaned_text = re.sub(r"\n\s*\n+", "\n", cleaned_text)

    return cleaned_text


def process_documents(docs):
    """Process and clean documents."""
    for i, doc in enumerate(docs):
        try:
            print(f"Processing document {i+1}: {doc.metadata.get('source', 'unknown')}")
            doc.page_content = clean_code_blocks(doc.page_content)
        except Exception as e:
            print(
                f"Error processing document {i+1} - {doc.metadata.get('source', 'unknown')}: {e}"
            )
            continue
    return docs


def main():
    try:
        local_loader = DirectoryLoader(
            "repos/next.js/docs",
            glob=["**/*.mdx", "**/*.md"],
            show_progress=True,
        )
        docs = local_loader.load()
        print(f"Length of the loaded documents: {len(docs)}")

        cleaned_docs = process_documents(docs)
        print(f"Length of the cleaned documents: {len(cleaned_docs)}")

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(cleaned_docs)
        print(f"Length of the split documents: {len(doc_splits)}")

        connection = os.getenv("DATABASE_URL")
        collection_name = "nextjs_docs"

        vector_store = PGVector(
            embeddings=OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL")),
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )

        vector_store.add_documents(doc_splits)
        print(f"Documents added to the vector store")

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
