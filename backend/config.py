import os
from dotenv import load_dotenv

# Load the .env file so os.environ can see our secrets
load_dotenv()

# --- Groq Settings ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Qdrant Settings ---
# Using local mode — runs inside Python, no Docker or server needed
# Data is persisted to a local folder so it survives restarts
QDRANT_MODE = "local"
QDRANT_LOCAL_PATH = os.path.join(os.path.dirname(__file__), "data", "qdrant_storage")
QDRANT_COLLECTION = "neurops_reviews"

# --- Embedding Settings ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# --- Data Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "olist")
REVIEWS_FILE = os.path.join(DATA_DIR, "olist_order_reviews_dataset.csv")
ORDERS_FILE = os.path.join(DATA_DIR, "olist_orders_dataset.csv")
CUSTOMERS_FILE = os.path.join(DATA_DIR, "olist_customers_dataset.csv")
ORDER_ITEMS_FILE = os.path.join(DATA_DIR, "olist_order_items_dataset.csv")
PAYMENTS_FILE = os.path.join(DATA_DIR, "olist_order_payments_dataset.csv")
PRODUCTS_FILE = os.path.join(DATA_DIR, "olist_products_dataset.csv")
CATEGORY_FILE = os.path.join(DATA_DIR, "product_category_name_translation.csv")