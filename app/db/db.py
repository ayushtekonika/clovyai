import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import os

# Global variable for the connection
load_dotenv()
db_connection = None

def create_connection():
    global db_connection
    if db_connection is None or not db_connection.is_connected():
        try:
            db_connection = mysql.connector.connect(
                host=os.environ["DB_HOST"],  # Connecting via localhost due to port forwarding
                port=os.environ["DB_PORT"], 
                user=os.environ["DB_USER"],
                password=os.environ["DB_PASSWORD"],
                database=os.environ["DB_NAME"]
            )
            print("Connection to MySQL DB successful")
        except Error as e:
            print(f"The error '{e}' occurred")
            raise

def close_connection():
    global db_connection
    if db_connection and db_connection.is_connected():
        db_connection.close()
