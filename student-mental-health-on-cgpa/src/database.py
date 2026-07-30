import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Load environment variables from .env file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=env_path)

# MongoDB Environment Settings
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "student_mental_health_db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "predictions")

def check_mongo_connection(timeout_ms: int = 2000):
    """
    Test connectivity to MongoDB server.
    Returns: (is_connected: bool, status_message: str)
    """
    if not MONGO_URI:
        return False, "MONGO_URI not configured in .env"
    
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=timeout_ms)
        client.admin.command('ping')
        return True, f"Connected to MongoDB ({MONGO_DB_NAME}.{MONGO_COLLECTION})"
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        return False, f"MongoDB Connection Failed: {str(e)}"
    except Exception as e:
        return False, f"MongoDB Error: {str(e)}"

def get_mongo_collection():
    """Get MongoDB collection handle."""
    is_connected, _ = check_mongo_connection()
    if not is_connected:
        return None
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = client[MONGO_DB_NAME]
    return db[MONGO_COLLECTION]

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
