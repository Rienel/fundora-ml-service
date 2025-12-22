import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

class DatabaseConnection:
    def __init__(self):
        self.connection = None
        self.config = {
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT', 5432),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.config)
            print(" Database connection established")
            return self.connection
        except Exception as e:
            print(f" Database connection failed: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("Database connection closed")
    
    def execute_query(self, query, params=None):
        """Execute a query and return results"""
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute(query, params)
            results = cursor.fetchall()
            return results
        except Exception as e:
            print(f"Query execution error: {e}")
            raise
        finally:
            cursor.close()

# Test the connection
if __name__ == "__main__":
    db = DatabaseConnection()
    db.connect()
    
    # Test query
    result = db.execute_query("SELECT COUNT(*) FROM core_startup")
    print(f"Total startups in database: {result[0]['count']}")
    
    db.close()