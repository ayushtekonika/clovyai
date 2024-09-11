# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
from fastapi import FastAPI, HTTPException
from fastapi import APIRouter
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from pydantic import BaseModel
from app.optimis import get_summary

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Optimis instance")
    yield
    
# Load environment variables
load_dotenv()
app = FastAPI(lifespan=lifespan)
router = APIRouter(prefix="/api/v1")
    
class QueryModel(BaseModel):
    query: str
    patientID: str

# Define your routes here
@router.get("/health")
async def health():
    return {"status": "ok"}

@router.post("/summary")
def summary(query_model: QueryModel):
    query = query_model.query
    patientID = query_model.patientID
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body")
    if not patientID:
        raise HTTPException(status_code=400, detail="Missing 'patientID' in request body")

    try:
        response = get_summary(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the FastAPI app
app.include_router(router)
# add router prefix api/v1

# to run it locally
# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host='0.0.0.0', port=8000)