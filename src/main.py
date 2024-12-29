import re

import tiktoken
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.sitemap import SitemapLoader


def clean_code_blocks(text: str) -> str:
    """Remove extra whitespace from code blocks
    while preserving regular text."""
    # Pattern to match code blocks with all possible metadata
    pattern = r'```(?:.*?)\n([\s\S]*?)```'

    def clean_block(match):
        code = match.group(1)
        # Remove empty lines and normalize whitespace in code
        cleaned = '\n'.join(line for line in code.split('\n') if line.strip())
        return f'```\n{cleaned}\n```'

    # Replace code blocks with cleaned versions
    cleaned_text = re.sub(pattern, clean_block, text, flags=re.DOTALL)

    # Remove multiple consecutive newlines from the entire document
    cleaned_text = re.sub(r'\n\s*\n+', '\n', cleaned_text)

    return cleaned_text


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    # Clean the text before counting tokens
    cleaned_text = clean_code_blocks(text)
    encoding = tiktoken.get_encoding("cl100k_base")
    # Allow endoftext token and disable special token checks
    return len(encoding.encode(cleaned_text, disallowed_special=()))


def process_documents(docs, source_name):
    """Process documents and return token statistics."""
    try:
        # Count tokens for the first document
        first_doc_tokens = count_tokens(docs[0].page_content)
        print(f"\n{source_name} - First document:")
        print(f"Source: {docs[0].metadata.get('source', 'N/A')}")
        print(f"Tokens: {first_doc_tokens}")

        # Count total tokens across all documents
        total_tokens = 0
        processed_docs = 0
        for i, doc in enumerate(docs):
            try:
                doc_tokens = count_tokens(doc.page_content)
                total_tokens += doc_tokens
                processed_docs += 1
                print(
                    f"Document {i}: {doc.metadata.get(
                        'source', 'N/A')} - {doc_tokens} tokens"
                )
            except Exception as e:
                print(f"Error counting tokens in document {i}: {e}")
                continue

        print(f"\n{source_name} Summary:")
        print(f"Total number of documents processed: {processed_docs}")
        print(f"Total number of tokens across all documents: {total_tokens}")
        print(
            f"Average tokens per document: {
                total_tokens/processed_docs if processed_docs > 0 else 0:.2f}"
        )
        return total_tokens, processed_docs

    except Exception as e:
        print(f"Error processing documents: {e}")
        return 0, 0


def main():
    try:
        local_loader = DirectoryLoader("repos/next.js/docs")
        local_docs = local_loader.load()
        local_tokens, local_docs_count = process_documents(
            local_docs, "Local Docs")

        sitemap_loader = SitemapLoader(
            web_path="https://nextjs.org/sitemap.xml",
            filter_urls=["https://nextjs.org/docs"],
        )
        sitemap_docs = sitemap_loader.load()
        sitemap_tokens, sitemap_docs_count = process_documents(
            sitemap_docs, "Sitemap Docs"
        )

        print("\nComparison:")
        print(
            f"Local docs count: {
                local_docs_count} | Total tokens: {local_tokens}"
        )
        print(
            f"Sitemap docs count: {
                sitemap_docs_count} | Total tokens: {sitemap_tokens}"
        )
        print(f"Difference in tokens: {abs(local_tokens - sitemap_tokens)}")

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
