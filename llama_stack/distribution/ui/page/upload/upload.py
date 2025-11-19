import streamlit as st
from llama_stack_client import RAGDocument
from llama_stack.distribution.ui.modules.api import llama_stack_api
from llama_stack.distribution.ui.modules.utils import data_url_from_file

def upload_page():
    """
    Page to upload documents and create a vector database for RAG.
    """
    st.title("📄 Upload")
    # File/Directory Upload Section
    st.subheader("Create Vector DB")
    # Let user select files to ingest
    uploaded_files = st.file_uploader(
        "Upload file(s) or directory",
        accept_multiple_files=True,
        type=["txt", "pdf", "doc", "docx"],  # supported file types
    )
    # Process uploaded files
    if uploaded_files:
        # Show upload success and prompt for DB name
        st.success(f"Successfully uploaded {len(uploaded_files)} files")
        vector_db_name = st.text_input(
            "Vector Database Name",
            value="rag_vector_db",
            help="Enter a unique identifier for this vector database",
        )
        if st.button("Create Vector Database"):
            # Convert uploaded files into RAGDocument instances
            documents = [
                RAGDocument(
                    document_id=uploaded_file.name,
                    content=data_url_from_file(uploaded_file),
                )
                for i, uploaded_file in enumerate(uploaded_files)
            ]

            # Determine provider for vector IO
            providers = llama_stack_api.client.providers.list()
            vector_io_provider = None
            for x in providers:
                if x.api == "vector_io":
                    vector_io_provider = x.provider_id

            # Create new vector store using modern API
            vs = llama_stack_api.client.vector_stores.create(
                name=vector_db_name,
                extra_body={
                    "embedding_model": "all-MiniLM-L6-v2",
                    "embedding_dimension": 384,
                    "provider_id": vector_io_provider,
                }
            )

            # Insert documents into the vector store using modern API
            for doc in documents:
                # Create a file from the document content
                from io import BytesIO
                file_content = BytesIO(doc.content.encode('utf-8'))
                file_content.name = f"{doc.document_id}.txt"
                
                # Upload file using the files API
                uploaded_file = llama_stack_api.client.files.create(
                    file=file_content,
                    purpose="assistants"
                )
                
                # Add the file to the vector store with chunking configuration
                llama_stack_api.client.vector_stores.files.create(
                    vector_store_id=vs.id,
                    file_id=uploaded_file.id,
                    chunking_strategy={
                        "type": "static",
                        "static": {
                            "max_chunk_size_tokens": 512,
                            "chunk_overlap_tokens": 50
                        }
                    }
                )
            st.success("Vector database created successfully!")
            # Reset form fields
            uploaded_files.clear()
            vector_db_name = ""
upload_page()