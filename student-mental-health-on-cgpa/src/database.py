import os
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# ---------------------------------------------------------------------------
# Credential Resolution — works in ALL environments:
#   1. Streamlit Cloud  → reads from st.secrets (toml-based secrets manager)
#   2. Local Dev        → reads from .env file via python-dotenv
# ---------------------------------------------------------------------------

def _get_mongo_settings():
    """
    Resolve MongoDB credentials with the following priority:
      1. Streamlit st.secrets  (when deployed on Streamlit Cloud)
      2. .env / OS environment (when running locally)
    Returns: (MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION)
    """
    # --- Try Streamlit secrets first (cloud deployment) ---
    try:
        import streamlit as st
        uri = st.secrets.get("MONGO_URI", None)
        if uri:
            db_name    = st.secrets.get("MONGO_DB_NAME", "student_mental_health_db")
            collection = st.secrets.get("MONGO_COLLECTION", "predictions")
            return uri, db_name, collection
    except Exception:
        pass  # streamlit not available or no secrets configured

    # --- Fall back to .env / environment variables (local) ---
    try:
        from dotenv import load_dotenv
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(BASE_DIR, '.env')
        load_dotenv(dotenv_path=env_path)
    except ImportError:
        pass

    uri        = os.getenv("MONGO_URI", "")
    db_name    = os.getenv("MONGO_DB_NAME", "student_mental_health_db")
    collection = os.getenv("MONGO_COLLECTION", "predictions")
    return uri, db_name, collection


def check_mongo_connection(timeout_ms: int = 5000):
    """
    Test connectivity to MongoDB server.
    Returns: (is_connected: bool, status_message: str)
    """
    uri, db_name, collection = _get_mongo_settings()

    if not uri:
        return False, "MONGO_URI not configured. Add it to .env (local) or Streamlit Secrets (cloud)."

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        client.admin.command('ping')
        return True, f"Connected to MongoDB Atlas ({db_name}.{collection})"
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        return False, f"MongoDB Connection Failed: {str(e)}"
    except Exception as e:
        return False, f"MongoDB Error: {str(e)}"


def get_mongo_collection():
    """Get MongoDB collection handle."""
    uri, db_name, col_name = _get_mongo_settings()
    is_connected, _ = check_mongo_connection()
    if not is_connected:
        return None
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = client[db_name]
    return db[col_name]


def log_prediction(prediction_record: dict):
    """
    Insert a prediction record into MongoDB.
    Returns: (success: bool, document_id_or_message: str)
    """
    collection = get_mongo_collection()
    if collection is None:
        return False, "MongoDB connection unavailable. Skipping database logging."

    try:
        record = prediction_record.copy()
        record['timestamp'] = datetime.utcnow().isoformat()
        result = collection.insert_one(record)
        return True, str(result.inserted_id)
    except Exception as e:
        return False, f"Failed to log prediction to MongoDB: {str(e)}"


def fetch_predictions_history(limit: int = 50):
    """Fetch recent prediction logs from MongoDB."""
    collection = get_mongo_collection()
    if collection is None:
        return []
    try:
        cursor = collection.find({}, {'_id': 0}).sort('timestamp', -1).limit(limit)
        return list(cursor)
    except Exception:
        return []
