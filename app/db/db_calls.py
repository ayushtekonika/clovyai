import mysql.connector
from fastapi import HTTPException

async def add_patient_summary(patientID, summary, db_connection):
    try:
        cursor = db_connection.cursor()

        # SQL query to insert the record
        insert_query = """
        INSERT INTO patient_summary (patientID, summary)
        VALUES (%s, %s)
        """
        cursor.execute(insert_query, (patientID, summary))

        # Commit the transaction
        db_connection.commit()

        # Close the cursor and connection
        cursor.close()
        db_connection.close()

        return {"message": "Patient summary added successfully."}

    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Database error: {err}")
    
async def get_patient_summary(patientID, db_connection):
    try:
        cursor = db_connection.cursor(dictionary=True)

        # SQL query to insert the record
        insert_query = """
        SELECT summary FROM patient_summary WHERE patientID = %s
        """
        cursor.execute(insert_query, (patientID, ))
        
        result = cursor.fetchone()

        cursor.close()

        if result:
            return result["summary"]
        else:
            raise HTTPException(status_code=404, detail="Patient not found")

    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Database error: {err}")