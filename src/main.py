import os
import re

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

framework_configurations = [
    {
        "framework": "Next.js",
        "glob": ["**/*.mdx", "**/*.md"],
        "directory": "repos/next.js/docs",
    }
]


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


def process_documents(docs, framework):
    """Process and clean documents while adding additional metadata."""
    for i, doc in enumerate(docs):
        try:
            # Clean the content
            doc.page_content = clean_code_blocks(doc.page_content)

            # Extract file path and name
            source_path = doc.metadata.get("source", "")

            # Add additional metadata
            doc.metadata.update(
                {
                    "file_name": os.path.basename(source_path),
                    "framework": framework,
                }
            )

            print(f"Processing document {i+1}: {doc.metadata}")
        except Exception as e:
            print(
                f"Error processing document {i+1} - {doc.metadata.get('source', 'unknown')}: {e}"
            )
            continue
    return docs


def main():
    try:
        connection_string = PGVector.connection_string_from_db_params(
            driver="psycopg",
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT")),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )

        print(connection_string)

        embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))

        for config in framework_configurations:
            print(f"\nProcessing {config['framework']} documentation...")

            # Load documents for the current framework
            local_loader = DirectoryLoader(
                config["directory"],
                glob=config["glob"],
                show_progress=True,
            )
            docs = local_loader.load()
            print(f"Length of the loaded documents: {len(docs)}")

            # Process and clean the documents
            cleaned_docs = process_documents(docs, config["framework"])
            print(f"Length of the cleaned documents: {len(cleaned_docs)}")

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=500, chunk_overlap=100
            )
            doc_splits = text_splitter.split_documents(cleaned_docs)
            print(f"Length of the split documents: {len(doc_splits)}")

            # Create or get vector store for this framework
            vector_store = PGVector(
                connection=connection_string,
                create_extension=True,
                embeddings=embeddings,
                collection_name="documentation",
                use_jsonb=True,
                # pre_delete_collection=True,  # This will recreate the table with correct schema
                collection_metadata={"framework": config["framework"]},
            )

            # Add documents to vector store
            vector_store.add_documents(doc_splits)
            print(f"Documents added to the vector store for {config['framework']}")

            # Example search
            results = vector_store.similarity_search(
                "server actions",
                filter={"framework": {"$in": [config["framework"]]}},
                k=1,
            )
            print(f"\nExample search results for {config['framework']}:")
            print(results)

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
