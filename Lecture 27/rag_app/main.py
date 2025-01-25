from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import traceback
from langgraph_flow import app 

# FastAPI app
fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. Replace with specific domains in production.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all HTTP headers
)

def process_content(result):
    content = []
    if len(result['documents']) > 1:
        for doc in result['documents']:
            content.append(doc.page_content)
    else:
        content.append(result['documents'].page_content)
    return content

# Pydantic models for inputs and outputs
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    generation: str
    documents: List[str]

# Define FastAPI routes
@fastapi_app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    inputs = {"question": request.question}

    try:
        result = app.invoke(inputs)
        final_generation = result['generation']
        final_documents = process_content(result)
        return QueryResponse(
                question=request.question,
                generation=final_generation,
                documents=final_documents,
            )

    except Exception as e:
        error_message = traceback.format_exc()
        print(f"Error during workflow execution:\n{error_message}")
        raise HTTPException(status_code=500, detail=str(e))
