"""
Tortoise ORM configuration for database connection.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "irrigation_db")
DB_USER = os.getenv("DB_USER", "irrigation_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# Tortoise ORM configuration
TORTOISE_ORM = {
    "connections": {
        "default": {
            "engine": "tortoise.backends.asyncpg",
            "credentials": {
                "host": DB_HOST,
                "port": DB_PORT,
                "user": DB_USER,
                "password": DB_PASSWORD,
                "database": DB_NAME,
                "min_size": 2,
                "max_size": 10,
            }
        }
    },
    "apps": {
        "models": {
            "models": ["api.database.models", "aerich.models"],
            "default_connection": "default",
        }
    },
    "use_tz": True,
    "timezone": "UTC",
}
