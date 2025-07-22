import os
import json
import hashlib
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("google_api_key") or ""


class FilingHashTracker:
    def __init__(self, path="indexed_filings.json"):
        self.path = path
        self.hashes = self.load_hashes()

    def load_hashes(self):
        if not os.path.exists(self.path):
            return set()
        with open(self.path, "r") as f:
            return set(json.load(f))

    def save_hashes(self):
        with open(self.path, "w") as f:
            json.dump(list(self.hashes), f)

    def get_hash(self, accession, form_type, filing_date):
        filing_str = f"{accession}_{form_type}_{filing_date}"
        return hashlib.md5(filing_str.encode()).hexdigest()

    def is_indexed(self, accession, form_type, filing_date):
        return self.get_hash(accession, form_type, filing_date) in self.hashes

    def mark_indexed(self, accession, form_type, filing_date):
        self.hashes.add(self.get_hash(accession, form_type, filing_date))
        self.save_hashes()


class FAISSManager:
    def __init__(self, index_path="faiss_index"):
        self.index_path = index_path
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        self.index = None
        self.hash_tracker = FilingHashTracker()
        self.load_index()

    def load_index(self):
        if os.path.exists(self.index_path):
            self.index = FAISS.load_local(
                self.index_path,
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
            print(f"Loaded FAISS index from {self.index_path}")
        else:
            self.index = None
            print("No existing FAISS index found.")

    def save_index(self):
        if self.index is not None:
            self.index.save_local(self.index_path)
            print(f"Saved FAISS index to {self.index_path}")

    def add_filings(self, filings, metadatas):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        new_documents = []

        for filing_text, metadata in zip(filings, metadatas):
            accession = metadata.get("accession")
            form_type = metadata.get("form_type")
            filing_date = metadata.get("filing_date")

            if self.hash_tracker.is_indexed(accession, form_type, filing_date):
                print(f"Skipping already indexed filing {accession}")
                continue

            # Chunk filing text
            chunks = splitter.split_text(filing_text)

            # Convert chunks to Documents with metadata
            docs = []
            for idx, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = idx
                docs.append(Document(page_content=chunk, metadata=chunk_metadata))

            new_documents.extend(docs)
            self.hash_tracker.mark_indexed(accession, form_type, filing_date)

        if not new_documents:
            print("No new filings to add.")
            return

        if self.index is None:
            print(f"Building new FAISS index with {len(new_documents)} chunks...")
            self.index = FAISS.from_documents(new_documents, self.embedding_model)
        else:
            print(f"Adding {len(new_documents)} new chunks to existing FAISS index...")
            self.index.add_documents(new_documents)

        self.save_index()

    def similarity_search(self, query, k=100):
        if self.index is None:
            raise RuntimeError("FAISS index is not built yet.")
        return self.index.similarity_search(query, k=k)

    def similarity_search_with_context(self, query, k=35, window=1):
        if self.index is None:
            raise RuntimeError("FAISS index is not built yet.")

        # Step 1: Get top-k most similar chunks
        top_docs = self.index.similarity_search(query, k=k)

        # Step 2: Load all chunks from docstore
        all_chunks = list(self.index.docstore._dict.values())  # type: ignore

        # Step 3: Organize chunks by filing id and index
        filing_chunks = {}
        for doc in all_chunks:
            filing_id = doc.metadata.get(
                "accession"
            )  # or use combined key if you prefer
            chunk_idx = doc.metadata.get("chunk_index")
            if filing_id is not None and chunk_idx is not None:
                filing_chunks.setdefault(filing_id, {})[chunk_idx] = doc

        # Step 4: For each top chunk, add neighbors within window
        expanded_chunks = {}
        for doc in top_docs:
            filing_id = doc.metadata.get("accession")
            chunk_idx = doc.metadata.get("chunk_index")
            if filing_id in filing_chunks:
                for offset in range(-window, window + 1):
                    neighbor_idx = chunk_idx + offset  # type: ignore
                    neighbor_doc = filing_chunks[filing_id].get(neighbor_idx)
                    if neighbor_doc:
                        key = (filing_id, neighbor_idx)
                        expanded_chunks[key] = neighbor_doc

        # Step 5: Return as list (sorted by filing and chunk index if desired)
        return sorted(
            expanded_chunks.values(),
            key=lambda d: (d.metadata.get("accession"), d.metadata.get("chunk_index")),
        )
