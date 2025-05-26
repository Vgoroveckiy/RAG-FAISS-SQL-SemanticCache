"""
main.py — Retrieval Augmented Generation (RAG) система для поиска и генерации ответов по ювелирным изделиям.

Функционал:
- Индексация документов (PDF, DOCX, TXT, JSON-каталоги) с помощью FAISS и LangChain.
- Хранение метаданных и текстов в SQLite.
- Семантический кэш вопросов/ответов.
- Интерактивный чат и тестовые вопросы.

Запуск:
    python main.py

Меню:
    1. Очистить данные (индексы FAISS и базу данных)
    2. Проиндексировать документы (из папки 'data')
    3. Запустить интерактивный чат
    4. Запустить тестовые вопросы
    0. Выйти

Требования:
    - Python 3.8+
    - Все зависимости из requirements.txt

Автор: [Вячеслав Горовецкий]
Дата: 2025
"""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
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
    raise ValueError(
        "OPENAI_API_KEY not found in .env file. Please create one and add your OPENAI_API_KEY."
    )
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
            metadata TEXT,
            faiss_chunk_ids TEXT -- Новое поле для хранения ID чанков FAISS
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
        faiss_chunk_ids: Optional[List[str]] = None,  # Новый аргумент
    ):
        """
        Добавляет или обновляет информацию о документе в базе данных.

        Args:
            file_path (str): Путь к файлу документа.
            file_hash (str): Хэш содержимого файла для отслеживания изменений.
            last_modified (float): Время последнего изменения файла (Unix timestamp).
            content (str): Извлеченное текстовое содержимое **всего** документа.
            metadata (dict): Дополнительные метаданные документа (например, источник).
            faiss_chunk_ids (Optional[List[str]]): Список ID чанков FAISS, относящихся к этому документу.
        """
        if faiss_chunk_ids is None:
            faiss_chunk_ids = []
        cursor = self.conn.cursor()
        cursor.execute(
            """
        INSERT OR REPLACE INTO documents (file_path, file_hash, last_modified, content, metadata, faiss_chunk_ids)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                file_path,
                file_hash,
                last_modified,
                content,
                json.dumps(metadata),
                json.dumps(faiss_chunk_ids),
            ),
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
            "SELECT file_hash, last_modified, content, metadata, faiss_chunk_ids FROM documents WHERE file_path = ?",
            (file_path,),
        )
        row = cursor.fetchone()
        if row:
            return {
                "file_hash": row[0],
                "last_modified": row[1],
                "content": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "faiss_chunk_ids": (
                    json.loads(row[4]) if row[4] else []
                ),  # Загружаем как список
            }
        return None

    def close(self):
        """Закрывает соединение с базой данных SQLite."""
        self.conn.close()


class VectorDatabase:
    """
    Управляет FAISS-векторными индексами для документов и кэша вопросов/ответов.
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

    def load_or_create(
        self, documents: Optional[List[Document]] = None, force_recreate: bool = False
    ):
        """
        Загружает или создает основной индекс FAISS.
        Если индекс существует, он загружается. Если нет, или force_recreate=True,
        он создается с нуля.

        Args:
            documents (Optional[List[Document]]): Список сегментов документов LangChain для индексации.
                                                  Используется только при создании нового индекса.
            force_recreate (bool): Если True, индекс будет пересоздан, даже если он существует.
                                   По умолчанию False.
        """
        if not force_recreate and os.path.exists(self.index_path):
            print(f"Загрузка существующего FAISS индекса из {self.index_path}...")
            self.db = FAISS.load_local(
                self.index_path, self.embeddings, allow_dangerous_deserialization=True
            )
        else:
            print(f"Создание нового FAISS индекса в {self.index_path}...")
            # Создаем пустой индекс, чтобы его можно было использовать для добавления/удаления
            dummy_doc = Document(page_content="initialization_dummy", metadata={})
            self.db = FAISS.from_documents([dummy_doc], self.embeddings)
            # Удаляем "пустой" документ, чтобы индекс был фактически пустым
            self.db.delete([self.db.index_to_docstore_id[0]])
            self.db.save_local(self.index_path)  # Сохраняем пустой индекс

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

    def add_to_cache(
        self, question: str, answer: str, sources: Optional[List[str]] = None
    ):
        """
        Добавляет вопрос и соответствующий ответ в кэш FAISS.

        Args:
            question (str): Запрос пользователя.
            answer (str): Сгенерированный ответ LLM.
            sources (Optional[List[str]]): Список источников (путей к файлам), на основе которых был сгенерирован ответ.
        """
        if not self.cache_db:
            self.load_or_create_cache()

        doc = Document(
            page_content=question,
            metadata={
                "answer": answer,
                "timestamp": datetime.now().isoformat(),
                "sources": (
                    sources if sources is not None else []
                ),  # Добавляем источники
            },
        )
        self.cache_db.add_documents([doc])
        self.cache_db.save_local(self.cache_path)

    def get_cached_answer(
        self,
        question: str,
        similarity_threshold: float = 0.1,
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

        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype=np.float32)

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

    def delete_documents(self, faiss_chunk_ids: List[str]):
        """
        Удаляет документы из FAISS индекса по их внутренним ID.

        Args:
            faiss_chunk_ids (List[str]): Список внутренних ID чанков FAISS для удаления.
        """
        if not self.db:
            print("FAISS индекс не инициализирован. Невозможно удалить документы.")
            return

        if faiss_chunk_ids:
            try:
                self.db.delete(faiss_chunk_ids)
                self.db.save_local(self.index_path)
                print(f"Удалено {len(faiss_chunk_ids)} старых чанков из FAISS.")
            except Exception as e:
                print(f"Ошибка при удалении чанков из FAISS: {e}")
        else:
            print("Нет FAISS chunk IDs для удаления.")

    def delete_cached_entries_by_source(self, source_file_name: str):
        """
        Удаляет все записи из кэша, которые ссылаются на данный исходный файл.
        Использует базовое имя файла (например, "catalog.json" или "document.pdf").

        Args:
            source_file_name (str): Базовое имя файла-источника для инвалидации кэша.
        """
        if not self.cache_db:
            print(
                "Кэш FAISS не инициализирован. Невозможно удалить кэшированные записи."
            )
            return

        if self.cache_db.index.ntotal == 0:
            print("Кэш пуст, нечего удалять.")
            return

        ids_to_delete = []
        for doc_id in list(
            self.cache_db.docstore._dict.keys()
        ):  # Iterate over docstore IDs
            doc = self.cache_db.docstore.search(doc_id)
            if doc and doc.metadata.get("sources"):
                # Проверяем, есть ли source_file_name в списке источников кэшированного ответа
                # Источники в метаданных - это полные пути (или file_path#index),
                # но мы их храним в add_to_cache как базовые имена, чтобы совпасть здесь.
                if source_file_name in doc.metadata["sources"]:
                    ids_to_delete.append(doc_id)

        if ids_to_delete:
            try:
                self.cache_db.delete(ids_to_delete)
                self.cache_db.save_local(self.cache_path)
                print(
                    f"Удалено {len(ids_to_delete)} кэшированных записей, связанных с источником '{source_file_name}'."
                )
            except Exception as e:
                print(f"Ошибка при удалении кэшированных записей: {e}")
        else:
            print(
                f"Не найдено кэшированных записей, связанных с источником '{source_file_name}'."
            )


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


def update_document_in_faiss(
    doc_storage: DocumentStorage,
    vector_db: VectorDatabase,
    file_path: str,
    full_text_content: str,
    metadata: dict,
    current_file_hash: str,
    current_last_modified: float,
    stored_doc_data: Optional[Dict[str, Any]],
) -> List[Document]:
    """
    Обновляет документ в FAISS: удаляет старые чанки и добавляет новые.
    Сохраняет новые FAISS IDs в DocumentStorage.
    Возвращает список новых чанков.

    Args:
        doc_storage (DocumentStorage): Объект хранилища документов.
        vector_db (VectorDatabase): Объект векторной базы данных.
        file_path (str): Уникальный путь к документу (например, "data/catalog.json#0").
        full_text_content (str): Полный текст документа.
        metadata (dict): Метаданные документа.
        current_file_hash (str): Хэш текущего содержимого файла.
        current_last_modified (float): Время последнего изменения файла.
        stored_doc_data (Optional[Dict[str, Any]]): Данные о документе из БД, если есть.

    Returns:
        List[Document]: Список новых чанков для индексации.
    """
    # Извлекаем базовое имя файла для инвалидации кэша
    base_file_name_for_cache_invalidation = os.path.basename(file_path.split("#")[0])

    if stored_doc_data and stored_doc_data.get("faiss_chunk_ids"):
        # Удаляем старые чанки из FAISS-индекса документов
        vector_db.delete_documents(stored_doc_data["faiss_chunk_ids"])
        print(f"Удалены старые FAISS чанки для {os.path.basename(file_path)}.")

    # **Инвалидация семантического кэша, связанного с этим источником**
    print(
        f"Инвалидация кэша вопросов/ответов, связанных с '{base_file_name_for_cache_invalidation}'..."
    )
    vector_db.delete_cached_entries_by_source(base_file_name_for_cache_invalidation)

    # Создаем новые чанки
    new_chunks = text_splitter.create_documents(
        [full_text_content], metadatas=[metadata]
    )

    new_faiss_chunk_ids = []
    if new_chunks:
        if not vector_db.db:
            vector_db.load_or_create()  # Убедимся, что индекс загружен или создан пустым
        new_faiss_chunk_ids = vector_db.db.add_documents(new_chunks)
        vector_db.db.save_local(vector_db.index_path)
        print(f"Добавлено {len(new_faiss_chunk_ids)} новых чанков в FAISS.")
    else:
        print(f"Нет новых чанков для добавления для {os.path.basename(file_path)}.")

    # Обновляем информацию о документе в SQL DB с новыми FAISS IDs
    doc_storage.add_document(
        file_path=file_path,
        file_hash=current_file_hash,
        last_modified=current_last_modified,
        content=full_text_content,
        metadata=metadata,
        faiss_chunk_ids=new_faiss_chunk_ids,
    )
    return new_chunks


def process_catalog_json(
    file_path: str, doc_storage: DocumentStorage, vector_db: VectorDatabase
) -> List[Document]:
    """
    Читает JSON-файл каталога, преобразует каждую запись в LangChain Document
    и сохраняет полный текст записи в SQL DB, а затем сегментирует для FAISS.
    Реализовано точечное обновление.

    Args:
        file_path (str): Путь к JSON-файлу каталога.
        doc_storage (DocumentStorage): Объект хранилища документов для сохранения метаданных.
        vector_db (VectorDatabase): Объект векторной базы данных для обновления FAISS.

    Returns:
        List[Document]: Список объектов LangChain Document (сегментов) для индексации.
    """
    documents_for_faiss = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            catalog_data = json.load(f)

        for i, item in enumerate(catalog_data):
            # Создаем текстовое представление для каждого элемента каталога
            item_text_content = (
                f"Название: {item.get('name', 'N/A')}\n"
                f"Описание: {item.get('description', 'N/A')}\n"
                f"Использование: {item.get('usage', 'N/A')}\n"
                f"Цена: {item.get('price', 'N/A')}\n"
                f"URL: {item.get('url', 'N/A')}"
            )

            # Создаем уникальный file_path для каждого элемента каталога
            unique_item_path = f"{file_path}#{i}"

            # Метаданные для каждого элемента каталога
            metadata = {
                "source": os.path.basename(file_path),
                "item_name": item.get("name", "N/A"),
                "item_url": item.get("url", "N/A"),
                "index_in_catalog": i,
            }

            file_hash = hashlib.sha256(item_text_content.encode()).hexdigest()
            last_modified = os.path.getmtime(file_path)

            stored_item = doc_storage.get_document(unique_item_path)

            if (
                stored_item
                and stored_item["file_hash"] == file_hash
                and stored_item["last_modified"] >= last_modified
            ):
                # Если элемент не изменился, используем кэшированный текст и чанки
                full_text_from_db = stored_item["content"]
                chunks = text_splitter.create_documents(
                    [full_text_from_db], metadatas=[metadata]
                )
                documents_for_faiss.extend(chunks)
                print(
                    f"Используется кэшированный элемент каталога: {item.get('name', 'N/A')} из {os.path.basename(file_path)}"
                )
            else:
                # Элемент новый или изменился, обновляем его в FAISS и SQL DB
                print(
                    f"Обработка нового/измененного элемента каталога: {item.get('name', 'N/A')} из {os.path.basename(file_path)}"
                )
                new_chunks_for_faiss = update_document_in_faiss(
                    doc_storage,
                    vector_db,
                    unique_item_path,
                    item_text_content,
                    metadata,
                    file_hash,
                    last_modified,
                    stored_item,
                )
                documents_for_faiss.extend(new_chunks_for_faiss)

        return documents_for_faiss

    except Exception as e:
        print(f"Ошибка при обработке файла каталога {file_path}: {str(e)}")
        return []


def parse_files(
    directory: str, doc_storage: DocumentStorage, vector_db: VectorDatabase
) -> List[Document]:
    """
    Парсит файлы из заданной директории, извлекает полный текст, сохраняет его в SQL DB,
    и затем сегментирует для индексации в FAISS. Реализовано точечное обновление.

    Args:
        directory (str): Путь к директории с файлами для обработки.
        doc_storage (DocumentStorage): Объект хранилища документов для сохранения метаданных.
        vector_db (VectorDatabase): Объект векторной базы данных для обновления FAISS.

    Returns:
        List[Document]: Список объектов LangChain Document (сегментов), представляющих содержимое файлов для FAISS.
    """
    all_chunks = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.lower().endswith(".json"):
            json_chunks = process_catalog_json(file_path, doc_storage, vector_db)
            all_chunks.extend(json_chunks)
            continue

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
                # Документ новый или изменился
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

                    print(f"Обработка нового/измененного документа: {filename}.")
                    new_chunks_for_faiss = update_document_in_faiss(
                        doc_storage,
                        vector_db,
                        file_path,
                        full_text_from_file,
                        file_metadata,
                        file_hash,
                        last_modified,
                        stored_doc,
                    )
                    all_chunks.extend(new_chunks_for_faiss)

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
        # Инициализация происходит в run() или в конкретных функциях,
        # чтобы иметь контроль над моментом загрузки/создания индексов.

    def initialize(self):
        """
        Инициализирует компоненты системы: загружает/обрабатывает документы и настраивает цепочки LangChain.
        """
        print("Инициализация RAG системы...")
        # Убедимся, что FAISS индекс инициализирован (загружен или создан пустым)
        self.vector_db.load_or_create()
        self.vector_db.load_or_create_cache()

        # Парсинг файлов с возможностью точечного обновления
        documents = parse_files(self.config.INPUT_DIR, self.doc_storage, self.vector_db)

        if not documents and self.vector_db.db.index.ntotal == 0:
            print(
                "ВНИМАНИЕ: Нет документов для индексации и FAISS индекс пуст. RAG система может работать неоптимально."
            )
        else:
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
            template=prompt_template,
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
            self.qa_chain = LLMChain(
                llm=self.llm,
                prompt=PromptTemplate(
                    input_variables=["input"], template="Вопрос: {input}\nОтвет:"
                ),
            )
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

        if self.retriever is None or (
            self.vector_db.db and self.vector_db.db.index.ntotal == 0
        ):
            print("Используется простая LLM-цепочка (нет документов для ретривала).")
            result = self.llm.invoke(f"Вопрос: {question}\nОтвет:")
            answer = result.content
            print("Ответ сгенерирован LLM без использования документов.")
            if use_cache:
                # В этом случае источников нет, т.к. LLM ответила без ретривала
                self.vector_db.add_to_cache(question, answer, sources=[])
                print("Ответ добавлен в кэш.")
            return answer

        # Использование create_retrieval_chain
        result = self.qa_chain.invoke({"input": question})
        answer = result.get("answer", "Не удалось сгенерировать ответ.")

        retrieved_sources = []
        if "context" in result:
            retrieved_docs = result["context"]
            print(f"Найдено {len(retrieved_docs)} релевантных документов.")
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get("source", "N/A")
                # Для JSON-файлов мы используем уникальный путь: data/catalog.json#0
                # Нужно извлечь только базовое имя файла для кэша: catalog.json
                if "#" in source:  # Это элемент из JSON-каталога
                    base_source = os.path.basename(source.split("#")[0])
                else:  # Это обычный файл
                    base_source = os.path.basename(source)

                # Добавляем только уникальные базовые имена источников
                if base_source not in retrieved_sources:
                    retrieved_sources.append(base_source)

                print(
                    f"--- Документ {i+1} (Источник: {doc.metadata.get('source', 'N/A')}, Item: {doc.metadata.get('item_name', 'N/A')}) ---"
                )
                print(f"Содержимое (часть): {doc.page_content[:500]}...")
                print("---------------------------------------")

        print("Ответ сгенерирован LLM с использованием документов.")

        if use_cache:
            # Передаем список использованных базовых имен источников в кэш
            self.vector_db.add_to_cache(question, answer, sources=retrieved_sources)
            print("Ответ добавлен в кэш.")

        return answer

    def close(self):
        """Закрывает соединения с базами данных."""
        self.doc_storage.close()
        print("Соединение с базой данных документов закрыто.")


# --- Функции для консольного приложения ---


def clean_data():
    """Очищает существующие индексы FAISS и базу данных SQLite."""
    print("\n--- Очистка данных ---")
    print("Выполняется ОЧИСТКА старых индексов и базы данных...")
    if os.path.exists(config.FAISS_INDEX_PATH):
        shutil.rmtree(config.FAISS_INDEX_PATH)
        print(f"Удалена папка FAISS индекса: {config.FAISS_INDEX_PATH}")
    if os.path.exists(config.CACHE_INDEX_PATH):
        shutil.rmtree(config.CACHE_INDEX_PATH)
        print(f"Удалена папка кэша FAISS: {config.CACHE_INDEX_PATH}")
    if os.path.exists(config.SQL_DB_PATH):
        os.remove(config.SQL_DB_PATH)
        print(f"Удалена база данных SQLite: {config.SQL_DB_PATH}")
    print("Очистка завершена.")
    input("\nНажмите Enter для продолжения...")


def run_indexing():
    """Запускает процесс индексации документов."""
    print("\n--- Запуск индексации документов ---")
    rag_system = RAGSystem(config)
    rag_system.initialize()
    rag_system.close()
    print("Индексация завершена.")
    input("\nНажмите Enter для продолжения...")


def run_interactive_chat():
    """Запускает интерактивный режим чата."""
    print("\n--- Интерактивный чат ---")
    print("Запуск интерактивного чата. Для выхода введите 'выход' или 'exit'.")
    rag_system = RAGSystem(config)
    rag_system.initialize()  # Инициализируем систему перед чатом

    while True:
        question = input("\nВаш вопрос: ")
        if question.lower() in ("выход", "exit"):
            break
        answer = rag_system.query(question)
        print(f"Ответ: {answer}")

    rag_system.close()
    print("Интерактивный чат завершен.")
    input("\nНажмите Enter для продолжения...")


def run_test_questions():
    """Запускает прогон с предопределенными тестовыми вопросами."""
    print("\n--- Прогон тестовых вопросов ---")
    rag_system = RAGSystem(config)
    rag_system.initialize()  # Инициализируем систему перед тестами

    questions = [
        "Какие серьги подойдут для вечернего мероприятия?",
        "Посоветуйте украшения для свадьбы",
        "Какие есть золотые кольца?",
        "Сколько стоит Колье с сапфирами?",
        "Как ухаживать за жемчугом?",
        "Опишите Серебряные серьги с аметистами.",
        "Что такое Печатка с ониксом?",
        "Какие украшения есть для мужчин?",
        "Есть ли у вас что-то с бриллиантами?",
    ]

    for i, question in enumerate(questions):
        print(f"\n--- Тестовый вопрос {i+1}/{len(questions)} ---")
        answer = rag_system.query(question)
        print(f"Ответ: {answer}\n")
        # Добавляем паузу, чтобы пользователь мог прочитать ответ
        if i < len(questions) - 1:
            input("Нажмите Enter, чтобы перейти к следующему вопросу...")

    rag_system.close()
    print("Прогон тестовых вопросов завершен.")
    input("\nНажмите Enter для продолжения...")


def display_menu():
    """Отображает главное меню приложения."""
    print("\n" + "=" * 50)
    print("          Система RAG для ювелирных украшений")
    print("=" * 50)
    print("1. Очистить данные (индексы FAISS и базу данных)")
    print("2. Проиндексировать документы (из папки 'data')")
    print("3. Запустить интерактивный чат")
    print("4. Запустить тестовые вопросы")
    print("0. Выйти")
    print("=" * 50)


def main():
    """Главная функция консольного приложения."""
    while True:
        display_menu()
        choice = input("Выберите опцию: ")

        if choice == "1":
            clean_data()
        elif choice == "2":
            run_indexing()
        elif choice == "3":
            run_interactive_chat()
        elif choice == "4":
            run_test_questions()
        elif choice == "0":
            print("Выход из приложения. До свидания!")
            break
        else:
            print("Неверный ввод. Пожалуйста, выберите число от 0 до 4.")


if __name__ == "__main__":
    main()
