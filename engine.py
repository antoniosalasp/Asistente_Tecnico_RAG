import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.memory.buffer_window import ConversationBufferWindowMemory
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

class OilGasGeminiBot:
    """Ingeniero experto en Oil & Gas usando Google Gemini."""

    def __init__(self, language: str = "Español", api_key: Optional[str] = None):
        self.language = language
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.llm = None
        self.search = DuckDuckGoSearchRun()
        
        # Memoria de ventana: guarda exactamente los últimos 7 intercambios
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=7,
            return_messages=True,
        )
        
        self.vector_store: Optional[Chroma] = None

    def _get_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "No se encontró la clave de Gemini. Defina GOOGLE_API_KEY o GEMINI_API_KEY en el entorno."
            )
        return self.api_key

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        if self.llm is None:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.2,
                api_key=self._get_api_key(),
            )
        return self.llm

    def ingest_files(self, file_paths: List[str]) -> str:
        """Carga documentos y crea el vector store en memoria."""
        documents = []
        for path in file_paths:
            loader = PyPDFLoader(path) if path.endswith(".pdf") else TextLoader(path)
            documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(documents)
        
        # Embeddings específicos de Google
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            api_key=self._get_api_key(),
        )
        
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
        )
        return "Conocimiento de ingeniería actualizado."

    def ask(self, question: str) -> str:
        if not question.strip():
            raise ValueError("La pregunta es requerida.")

        prompt_base = (
            f"Actúa como Ingeniero experto en Oil and Gas. Responde en {self.language}. "
            "Respuesta técnica, específica y menor a 300 caracteres."
        )

        if self.vector_store:
            # Recupera documentos relevantes y usa el LLM para generar la respuesta
            retriever = self.vector_store.as_retriever()
            docs = retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)
            prompt = (
                f"Actúa como Ingeniero experto en Oil and Gas. Responde en {self.language}. "
                "Respuesta técnica, específica y menor a 300 caracteres."
                f"\n\nContexto:\n{context}\n\nPregunta: {question}"
            )
            response = self._get_llm().invoke(prompt)
        else:
            # Fallback a búsqueda web
            search_data = self.search.run(question)
            response = self._get_llm().invoke(
                f"{prompt_base} Contexto web: {search_data} Pregunta: {question}"
            )

        return response.content[:300]