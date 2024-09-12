import mysql.connector
from mysql.connector import Error

# Global variable for the connection
db_connection = None

def create_connection():
    global db_connection
    if db_connection is None or not db_connection.is_connected():
        try:
            db_connection = mysql.connector.connect(
                host="localhost",  # Connecting via localhost due to port forwarding
                port=3310, 
                user="optimis",
                password="Admin@123",
                database="optimis_development"
            )
            print("Connection to MySQL DB successful")
        except Error as e:
            print(f"The error '{e}' occurred")
            raise

def close_connection():
    global db_connection
    if db_connection and db_connection.is_connected():
        db_connection.close()
