import os
import json
import hashlib
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
    def __init__(self, index_dir="faiss_indexes"):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        self.indexes = {}  # Map company ticker -> FAISS index instance
        self.hash_tracker = FilingHashTracker()

    def load_index(self, company):
        index_path = os.path.join(self.index_dir, f"{company}.faiss")
        if os.path.exists(index_path):
            self.indexes[company] = FAISS.load_local(
                index_path,
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
            print(f"Loaded FAISS index for {company} from {index_path}")
        else:
            self.indexes[company] = None
            print(f"No existing FAISS index found for {company}")

    def save_index(self, company):
        index_path = os.path.join(self.index_dir, f"{company}.faiss")
        if self.indexes.get(company) is not None:
            self.indexes[company].save_local(index_path)
            print(f"Saved FAISS index for {company} to {index_path}")

    def add_filings(self, filings, metadatas, company, isPressRelease=False):
        if company not in self.indexes:
            self.load_index(company)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        new_documents = []
        for filing_text, metadata in zip(filings, metadatas):
            ticker = metadata.get("ticker") or company

            # Determine hash keys based on press release or not
            if not isPressRelease:
                accession = metadata.get("accession")
                form_type = metadata.get("form_type")
                filing_date = metadata.get("filing_date")

                if self.hash_tracker.is_indexed(accession, form_type, filing_date):
                    print(f"Skipping already indexed filing {accession} for {company}")
                    continue
            else:
                form_type = "press release"
                filing_date = metadata.get("filing_date")

                if self.hash_tracker.is_indexed(ticker, form_type, filing_date):
                    print(f"Skipping already indexed press release for {company}")
                    continue

            chunks = splitter.split_text(filing_text)
            docs = []
            for idx, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = idx
                docs.append(Document(page_content=chunk, metadata=chunk_metadata))

            new_documents.extend(docs)

            # Mark indexed after processing
            if not isPressRelease:
                self.hash_tracker.mark_indexed(accession, form_type, filing_date)
            else:
                self.hash_tracker.mark_indexed(ticker, form_type, filing_date)

        if not new_documents:
            print(f"No new filings to add for {company}.")
            return

        if self.indexes[company] is None:
            print(
                f"Building new FAISS index for {company} with {len(new_documents)} chunks..."
            )
            self.indexes[company] = FAISS.from_documents(
                new_documents, self.embedding_model
            )
        else:
            print(
                f"Adding {len(new_documents)} chunks to existing FAISS index for {company}..."
            )
            self.indexes[company].add_documents(new_documents)

        self.save_index(company)

    def similarity_search(self, company, query, k=100):
        if company not in self.indexes or self.indexes[company] is None:
            raise RuntimeError(f"FAISS index for {company} not loaded.")
        return self.indexes[company].similarity_search(query, k=k)

    def similarity_search_with_context(self, company, query, k=35, window=1):
        if company not in self.indexes or self.indexes[company] is None:
            raise RuntimeError(f"FAISS index for {company} not loaded.")

        # Step 1: Get top-k most similar chunks from that company's index
        top_docs = self.indexes[company].similarity_search(query, k=k)

        # Step 2: Load all chunks from docstore of that index
        all_chunks = list(self.indexes[company].docstore._dict.values())  # type: ignore

        # Step 3: Organize chunks by filing id and index
        filing_chunks = {}
        for doc in all_chunks:
            filing_id = doc.metadata.get("accession")
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

        print(f"Found {len(expanded_chunks)} chunks for company {company}")

        # Step 5: Return as list (sorted by filing and chunk index)
        return sorted(
            expanded_chunks.values(),
            key=lambda d: (d.metadata.get("accession"), d.metadata.get("chunk_index")),
        )
