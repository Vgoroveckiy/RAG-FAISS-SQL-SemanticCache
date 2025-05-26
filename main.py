import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from unstructured.partition.auto import partition

# Загрузка переменных окружения
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")
os.environ["OPENAI_API_KEY"] = api_key


class Config:
    """
    Конфигурация системы RAG.

    Определяет пути к файлам и директориям, используемые модели,
    и параметры для обработки и генерации ответов.
    """

    INPUT_DIR: str = "data"
    """Директория для исходных документов, которые будут индексироваться."""
    FAISS_INDEX_PATH: str = "./faiss_index"
    """Путь для сохранения основного индекса FAISS, содержащего эмбеддинги документов."""
    SQL_DB_PATH: str = "./documents.db"
    """Путь к файлу базы данных SQLite для хранения метаданных и содержимого документов."""
    CACHE_INDEX_PATH: str = "./cache_index"
    """Путь для сохранения индекса FAISS, используемого для кэширования вопросов и ответов."""
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    """Название модели эмбеддингов HuggingFace для создания векторных представлений текста."""
    LLM_MODEL: str = "gpt-4o-mini"
    """Название модели LLM (Large Language Model) от OpenAI для генерации ответов."""
    LLM_TEMPERATURE: float = 0.5
    """
    Температура LLM.
    Более высокие значения (например, 0.7) делают ответы более случайными и креативными,
    более низкие значения (например, 0.2) делают их более сфокусированными и детерминированными.
    """
    SUPPORTED_EXTENSIONS: tuple = (".pdf", ".docx", ".txt", ".json")  # Добавляем .json
    """Кортеж поддерживаемых расширений файлов для обработки."""


# Инициализация объекта конфигурации
config = Config()

# --- ВРЕМЕННЫЙ КОД ДЛЯ ОЧИСТКИ ИНДЕКСОВ И БД ---
# УДАЛИТЕ ЭТИ СТРОКИ ПОСЛЕ ПЕРВОГО УСПЕШНОГО ЗАПУСКА,
# ЧТОБЫ КЭШИРОВАНИЕ ДОКУМЕНТОВ И РАБОТА С FAISS РАБОТАЛИ КОРРЕКТНО БЕЗ ПЕРЕИНДЕКСАЦИИ КАЖДЫЙ РАЗ!
# print("Выполняется ОЧИСТКА старых индексов и базы данных...")
# if os.path.exists(config.FAISS_INDEX_PATH):
#     shutil.rmtree(config.FAISS_INDEX_PATH)
#     print(f"Удалена папка FAISS индекса: {config.FAISS_INDEX_PATH}")
# if os.path.exists(config.CACHE_INDEX_PATH):
#     shutil.rmtree(config.CACHE_INDEX_PATH)
#     print(f"Удалена папка кэша FAISS: {config.CACHE_INDEX_PATH}")
# if os.path.exists(config.SQL_DB_PATH):
#     os.remove(config.SQL_DB_PATH)
#     print(f"Удалена база данных SQLite: {config.SQL_DB_PATH}")
# print("Очистка завершена.")
# --- КОНЕЦ ВРЕМЕННОГО КОДА ---


# Создание директории для входных документов, если она не существует
os.makedirs(config.INPUT_DIR, exist_ok=True)

# Инициализация эмбеддингов и LLM на основе конфигурации
embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
llm = ChatOpenAI(model_name=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)

# Инициализация Text Splitter один раз (ГЛОБАЛЬНО)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)


class DocumentStorage:
    """
    Класс для управления хранением метаданных и содержимого документов в базе данных SQLite.

    Позволяет отслеживать изменения файлов и избегать повторной обработки.
    Сохраняет полный текст файла в поле 'content'.
    """

    def __init__(self, db_path: str):
        """
        Инициализирует подключение к базе данных SQLite.

        Args:
            db_path (str): Путь к файлу базы данных SQLite.
        """
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        """Инициализирует схему базы данных, создавая таблицу 'documents' при необходимости."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            file_hash TEXT,
            last_modified REAL,
            content TEXT,
            metadata TEXT
        )
        """
        )
        self.conn.commit()

    def add_document(
        self,
        file_path: str,
        file_hash: str,
        last_modified: float,
        content: str,
        metadata: dict,
    ):
        """
        Добавляет или обновляет информацию о документе в базе данных.

        Args:
            file_path (str): Путь к файлу документа.
            file_hash (str): Хэш содержимого файла для отслеживания изменений.
            last_modified (float): Время последнего изменения файла (Unix timestamp).
            content (str): Извлеченное текстовое содержимое **всего** документа.
            metadata (dict): Дополнительные метаданные документа (например, источник).
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
        INSERT OR REPLACE INTO documents (file_path, file_hash, last_modified, content, metadata)
        VALUES (?, ?, ?, ?, ?)
        """,
            (file_path, file_hash, last_modified, content, json.dumps(metadata)),
        )
        self.conn.commit()

    def get_document(self, file_path: str) -> Optional[dict]:
        """
        Извлекает информацию о документе из базы данных по его пути.

        Args:
            file_path (str): Путь к файлу документа.

        Returns:
            Optional[dict]: Словарь с информацией о документе (file_hash, last_modified, content, metadata)
                            или None, если документ не найден.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT file_hash, last_modified, content, metadata FROM documents WHERE file_path = ?",
            (file_path,),
        )
        row = cursor.fetchone()
        if row:
            return {
                "file_hash": row[0],
                "last_modified": row[1],
                "content": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
            }
        return None

    def close(self):
        """Закрывает соединение с базой данных SQLite."""
        self.conn.close()


class VectorDatabase:
    """
    Класс для управления векторными базами данных FAISS.

    Используется для основного индекса документов и для кэша вопросов/ответов.
    """

    def __init__(
        self, index_path: str, cache_path: str, embeddings: HuggingFaceEmbeddings
    ):
        """
        Инициализирует пути к индексам и модель эмбеддингов.

        Args:
            index_path (str): Путь для сохранения основного индекса FAISS.
            cache_path (str): Путь для сохранения индекса FAISS кэша.
            embeddings (HuggingFaceEmbeddings): Модель эмбеддингов для векторизации текста.
        """
        self.index_path = index_path
        self.cache_path = cache_path
        self.embeddings = embeddings
        self.db: Optional[FAISS] = None
        """Основной FAISS индекс для документов."""
        self.cache_db: Optional[FAISS] = None
        """FAISS индекс для кэшированных вопросов и ответов."""

    def load_or_create(self, documents: List[Document], force_recreate: bool = False):
        """
        Загружает или создает основной индекс FAISS из списка сегментов документов.

        Args:
            documents (List[Document]): Список сегментов документов LangChain для индексации.
            force_recreate (bool): Если True, индекс будет пересоздан, даже если он существует.
                                   По умолчанию False.
        Raises:
            ValueError: Если documents пуст при создании нового индекса.
        """
        if not force_recreate and os.path.exists(self.index_path):
            print(f"Загрузка существующего FAISS индекса из {self.index_path}...")
            self.db = FAISS.load_local(
                self.index_path, self.embeddings, allow_dangerous_deserialization=True
            )
        else:
            if documents:
                print(f"Создание нового FAISS индекса в {self.index_path}...")
                self.db = FAISS.from_documents(documents, self.embeddings)
                self.db.save_local(self.index_path)
            else:
                raise ValueError("No documents provided to create FAISS index")

    def load_or_create_cache(self, force_recreate: bool = False):
        """
        Загружает или создает индекс кэша FAISS.

        Args:
            force_recreate (bool): Если True, кэш будет пересоздан, даже если он существует.
                                   По умолчанию False.
        """
        if not force_recreate and os.path.exists(self.cache_path):
            print(f"Загрузка существующего FAISS кэша из {self.cache_path}...")
            self.cache_db = FAISS.load_local(
                self.cache_path, self.embeddings, allow_dangerous_deserialization=True
            )
        else:
            print(f"Создание нового FAISS кэша в {self.cache_path}...")
            dummy_doc = Document(page_content="dummy", metadata={"answer": "dummy"})
            self.cache_db = FAISS.from_documents([dummy_doc], self.embeddings)
            self.cache_db.delete([self.cache_db.index_to_docstore_id[0]])
            self.cache_db.save_local(self.cache_path)

    def add_to_cache(self, question: str, answer: str):
        """
        Добавляет вопрос и соответствующий ответ в кэш FAISS.

        Args:
            question (str): Запрос пользователя.
            answer (str): Сгенерированный ответ LLM.
        """
        if not self.cache_db:
            self.load_or_create_cache()

        doc = Document(
            page_content=question,
            metadata={"answer": answer, "timestamp": datetime.now().isoformat()},
        )
        self.cache_db.add_documents([doc])
        self.cache_db.save_local(self.cache_path)

    def get_cached_answer(
        self,
        question: str,
        similarity_threshold: float = 0.1,  # Для L2 расстояния, чем меньше, тем лучше
    ) -> Optional[str]:
        """
        Пытается найти кэшированный ответ на вопрос, если существует достаточно похожий запрос.

        Args:
            question (str): Запрос пользователя.
            similarity_threshold (float): Порог расстояния (L2). Если расстояние найденного вопроса
                                         меньше или равно порогу, возвращается кэшированный ответ.

        Returns:
            Optional[str]: Кэшированный ответ, если найден и достаточно схож, иначе None.
        """
        if not self.cache_db:
            self.load_or_create_cache()

        if not self.cache_db.index.ntotal:
            return None

        query_embedding = self.embeddings.embed_query(question)

        # --- НОВОЕ ДОБАВЛЕНИЕ: Преобразование в NumPy массив ---
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        # --- КОНЕЦ НОВОГО ДОБАВЛЕНИЯ ---

        scores, indices = self.cache_db.index.search(query_embedding.reshape(1, -1), 1)

        if scores[0][0] <= similarity_threshold:
            doc_id = self.cache_db.index_to_docstore_id[indices[0][0]]
            doc = self.cache_db.docstore.search(doc_id)
            print(f"Кэш: Найдено совпадение с расстоянием L2 = {scores[0][0]:.4f}")
            return doc.metadata["answer"]
        else:
            print(
                f"Кэш: Ближайшее совпадение с расстоянием L2 = {scores[0][0]:.4f} (выше порога {similarity_threshold})."
            )
        return None


def get_file_hash(file_path: str) -> str:
    """
    Вычисляет SHA256-хэш содержимого файла.

    Используется для определения, изменился ли файл с момента последней обработки.

    Args:
        file_path (str): Путь к файлу.

    Returns:
        str: Шестнадцатеричное представление SHA256-хэша файла.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def process_catalog_json(
    file_path: str, doc_storage: DocumentStorage
) -> List[Document]:
    """
    Читает JSON-файл каталога, преобразует каждую запись в LangChain Document
    и сохраняет полный текст записи в SQL DB, а затем сегментирует для FAISS.

    Args:
        file_path (str): Путь к JSON-файлу каталога.
        doc_storage (DocumentStorage): Объект хранилища документов для сохранения метаданных.

    Returns:
        List[Document]: Список объектов LangChain Document (сегментов) для индексации.
    """
    documents_for_faiss = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            catalog_data = json.load(f)

        for i, item in enumerate(catalog_data):
            # Создаем текстовое представление для каждого элемента каталога
            # чтобы LLM мог его легко понять
            item_text_content = (
                f"Название: {item.get('name', 'N/A')}\n"
                f"Описание: {item.get('description', 'N/A')}\n"
                f"Использование: {item.get('usage', 'N/A')}\n"
                f"Цена: {item.get('price', 'N/A')}\n"
                f"URL: {item.get('url', 'N/A')}"
            )

            # Создаем уникальный file_path для каждого элемента каталога
            # Это важно, так как в SQL DB file_path UNIQUE
            unique_item_path = f"{file_path}#{i}"

            # Метаданные для каждого элемента каталога
            metadata = {
                "source": os.path.basename(file_path),
                "item_name": item.get("name", "N/A"),
                "item_url": item.get("url", "N/A"),
                "index_in_catalog": i,  # Полезно для отладки
            }

            # Проверяем, есть ли уже этот элемент в кэше SQL DB
            file_hash = hashlib.sha256(item_text_content.encode()).hexdigest()
            last_modified = os.path.getmtime(
                file_path
            )  # Используем время изменения файла каталога

            stored_item = doc_storage.get_document(unique_item_path)

            if (
                stored_item
                and stored_item["file_hash"] == file_hash
                and stored_item["last_modified"] >= last_modified
            ):
                # Если элемент не изменился, используем кэшированный текст
                full_text_from_db = stored_item["content"]
                chunks = text_splitter.create_documents(
                    [full_text_from_db], metadatas=[metadata]
                )
                documents_for_faiss.extend(chunks)
                print(
                    f"Используется кэшированный элемент каталога: {item.get('name', 'N/A')} из {os.path.basename(file_path)}"
                )
            else:
                # Сохраняем полный текст элемента в SQL DB
                doc_storage.add_document(
                    file_path=unique_item_path,
                    file_hash=file_hash,
                    last_modified=last_modified,
                    content=item_text_content,
                    metadata=metadata,
                )
                print(
                    f"Обработан новый/измененный элемент каталога: {item.get('name', 'N/A')} из {os.path.basename(file_path)}"
                )

                # Сегментируем текст элемента для FAISS
                chunks = text_splitter.create_documents(
                    [item_text_content], metadatas=[metadata]
                )
                documents_for_faiss.extend(chunks)

        return documents_for_faiss

    except Exception as e:
        print(f"Ошибка при обработке файла каталога {file_path}: {str(e)}")
        return []


def parse_files(directory: str, doc_storage: DocumentStorage) -> List[Document]:
    """
    Парсит файлы из заданной директории, извлекает полный текст, сохраняет его в SQL DB,
    и затем сегментирует для индексации в FAISS.

    Проверяет изменения файлов и использует кэшированные полные тексты при их отсутствии.

    Args:
        directory (str): Путь к директории с файлами для обработки.
        doc_storage (DocumentStorage): Объект хранилища документов для сохранения метаданных.

    Returns:
        List[Document]: Список объектов LangChain Document (сегментов), представляющих содержимое файлов для FAISS.
    """
    all_chunks = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.lower().endswith(".json"):
            # Обработка JSON-файлов каталога
            json_chunks = process_catalog_json(file_path, doc_storage)
            all_chunks.extend(json_chunks)
            continue  # Переходим к следующему файлу

        if filename.lower().endswith(
            (".pdf", ".docx", ".txt")
        ):  # Остальные поддерживаемые типы
            file_hash = get_file_hash(file_path)
            last_modified = os.path.getmtime(file_path)

            stored_doc = doc_storage.get_document(file_path)

            if (
                stored_doc
                and stored_doc["file_hash"] == file_hash
                and stored_doc["last_modified"] >= last_modified
            ):
                full_text_from_db = stored_doc["content"]
                metadata = stored_doc["metadata"]

                if full_text_from_db.strip():
                    chunks = text_splitter.create_documents(
                        [full_text_from_db], metadatas=[metadata]
                    )
                    all_chunks.extend(chunks)
                    print(
                        f"Используется кэшированный документ для: {filename}. Сегментов: {len(chunks)}"
                    )
                else:
                    print(
                        f"Внимание: Кэшированный текст для {filename} пуст. Повторная обработка."
                    )
            else:
                try:
                    elements = partition(filename=file_path)
                    full_text_from_file = "\n\n".join(
                        [e.text for e in elements if e.text and e.text.strip()]
                    )

                    if not full_text_from_file.strip():
                        print(
                            f"Внимание: Не удалось извлечь осмысленный текст из {filename}. Пропускаем."
                        )
                        continue

                    file_metadata = {"source": filename, "file_path": file_path}

                    doc_storage.add_document(
                        file_path=file_path,
                        file_hash=file_hash,
                        last_modified=last_modified,
                        content=full_text_from_file,
                        metadata=file_metadata,
                    )
                    print(f"Обработан новый/измененный документ: {filename}.")

                    chunks = text_splitter.create_documents(
                        [full_text_from_file], metadatas=[file_metadata]
                    )
                    all_chunks.extend(chunks)
                    print(f"Создано {len(chunks)} сегментов для индексации FAISS.")

                except Exception as e:
                    print(f"Ошибка при обработке файла {file_path}: {str(e)}")
    return all_chunks


class RAGSystem:
    """
    Основной класс системы Retrieval Augmented Generation (RAG).

    Оркестрирует работу с хранилищем документов, векторной базой и LLM для ответа на вопросы.
    """

    def __init__(self, config: Config):
        """
        Инициализирует систему RAG.

        Args:
            config (Config): Объект конфигурации системы.
        """
        self.config = config
        self.doc_storage = DocumentStorage(config.SQL_DB_PATH)
        self.vector_db = VectorDatabase(
            config.FAISS_INDEX_PATH, config.CACHE_INDEX_PATH, embeddings
        )
        self.llm = llm
        self.retriever: Any = None
        self.qa_chain: Any = None
        self.prompt: PromptTemplate = None
        self.initialize()

    def initialize(self):
        """
        Инициализирует компоненты системы: загружает/обрабатывает документы и настраивает цепочки LangChain.
        """
        print("Инициализация RAG системы...")
        documents = parse_files(self.config.INPUT_DIR, self.doc_storage)

        if not documents:
            print(
                "ВНИМАНИЕ: Не удалось обработать ни одного документа для индексации. RAG система не будет работать корректно без данных."
            )

        self.vector_db.load_or_create(documents)
        self.vector_db.load_or_create_cache()
        print("Документы загружены и векторные базы инициализированы.")

        self._init_chains()
        print("Цепочки LangChain инициализированы.")

    def _init_chains(self):
        """Инициализирует промпт и цепочки LangChain для RAG."""
        prompt_template = """
        Ты — умный ассистент, специализирующийся на ювелирных украшениях.
        Ваши основные задачи:
        1. Отвечать на вопросы о ювелирных украшениях, их характеристиках, использовании и ценах по следующему контексту: {context}
        2. Помогать клиентам в выборе подходящих товаров.
        Ваша цель — предоставлять полезные, понятные и дружелюбные ответы.
        Если вы не знаете ответа, просто скажите: «Я не знаю». Не придумывайте информацию.
        При предложении товаров старайтесь быть конкретным и описывать, как товар может помочь.
        Если для ответа требуется больше информации, задавайте уточняющие вопросы.
        Вопрос: {input}
        """
        self.prompt = PromptTemplate(
            input_variables=["context", "input"],
            template=prompt_template,  # <--- УБЕДИТЕСЬ, ЧТО ЗДЕСЬ ['context', 'input']
        )

        if self.vector_db.db and self.vector_db.db.index.ntotal > 0:
            self.retriever = self.vector_db.db.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )

            from langchain.chains import create_retrieval_chain
            from langchain.chains.combine_documents import create_stuff_documents_chain

            combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)

            self.qa_chain = create_retrieval_chain(self.retriever, combine_docs_chain)

        else:
            print(
                "ВНИМАНИЕ: Основной FAISS индекс пуст. Невозможно инициализировать RetrievalQA. Будет использоваться простая LLM-цепочка."
            )
            # Для LLMChain тоже нужно изменить input_variables, если вы будете ее использовать
            # Если промпт всегда ожидает 'input' и 'context', то и здесь нужно поправить
            self.qa_chain = LLMChain(llm=self.llm, prompt=self.prompt)
            self.retriever = None

    def query(self, question: str, use_cache: bool = True) -> str:
        """
        Обрабатывает запрос пользователя, используя RAG-систему и кэш.
        """
        print(f"\n--- Обработка запроса: {question} ---")
        if use_cache:
            cached_answer = self.vector_db.get_cached_answer(question)
            if cached_answer:
                print("Ответ найден в кэше.")
                return cached_answer

        if self.retriever is None:
            print("Используется простая LLM-цепочка (нет документов для ретривала).")
            result = self.qa_chain.invoke({"query": question, "context": ""})
            answer = result.get(
                "text", result.get("answer", "Не удалось сгенерировать ответ.")
            )
            print("Ответ сгенерирован LLM без использования документов.")
            if use_cache:
                self.vector_db.add_to_cache(question, answer)
                print("Ответ добавлен в кэш.")
            return answer

        # При использовании create_retrieval_chain, вам не нужно
        # вызывать self.retriever.get_relevant_documents(question) вручную.
        # Цепочка create_retrieval_chain сама позаботится о ретривале
        # и передаче документов.

        # ИЗМЕНЕНИЕ ЗДЕСЬ: УДАЛЯЕМ МАНУАЛЬНЫЙ ВЫЗОВ РЕТРИВЕРА
        # retrieved_docs = self.retriever.get_relevant_documents(question)
        # if not retrieved_docs:
        #     print("Внимание: Ретривер не нашел релевантных документов для запроса.")
        #     answer = "Я не смог найти информацию по вашему запросу в доступных документах."
        #     if use_cache:
        #         self.vector_db.add_to_cache(question, answer)
        #     return answer

        # ИЗМЕНЕНИЕ ЗДЕСЬ: Вызов `invoke` для `create_retrieval_chain`
        # ему нужен только основной запрос, он сам вызовет ретривер
        result = self.qa_chain.invoke({"input": question})

        # `create_retrieval_chain` возвращает словарь, который обычно содержит 'answer' и 'context' (retrieved_documents)
        answer = result.get("answer", "Не удалось сгенерировать ответ.")

        # Дополнительно, вы можете вывести найденные документы, если нужно для отладки
        if "context" in result:
            retrieved_docs = result["context"]
            print(f"Найдено {len(retrieved_docs)} релевантных документов.")
            for i, doc in enumerate(retrieved_docs):
                print(
                    f"--- Документ {i+1} (Источник: {doc.metadata.get('source', 'N/A')}, Item: {doc.metadata.get('item_name', 'N/A')}) ---"
                )
                print(f"Содержимое (часть): {doc.page_content[:500]}...")
                print("---------------------------------------")

        print("Ответ сгенерирован LLM с использованием документов.")

        if use_cache:
            self.vector_db.add_to_cache(question, answer)
            print("Ответ добавлен в кэш.")

        return answer

    def close(self):
        """Закрывает соединения с базами данных."""
        self.doc_storage.close()
        print("Соединение с базой данных документов закрыто.")


if __name__ == "__main__":
    # Убедитесь, что catalog.json находится в папке 'data'
    # Например: data/catalog.json

    rag_system = RAGSystem(config)

    questions = [
        "Какие серьги подойдут для вечернего мероприятия?",
        "Посоветуйте украшения для свадьбы",
        "Какие есть золотые кольца?",
        "Сколько стоит Колье с сапфирами?",
        "Как ухаживать за жемчугом?",
        "Опишите Серебряные серьги с аметистами.",
        "Что такое Печатка с ониксом?",
    ]

    for question in questions:
        answer = rag_system.query(question)
        print(f"Ответ: {answer}\n")

    rag_system.close()
    print("\nRAG система завершила работу.")
