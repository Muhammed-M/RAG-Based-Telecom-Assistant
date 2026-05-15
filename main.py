from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests # Used to trigger your n8n webhook
from MyRAG_class import MyRAG

# 1. Initialize the FastAPI app and your RAG engine
app = FastAPI(title="NileTel Support API")
rag_engine = MyRAG()

# 2. Define the Data Models (Pydantic)
# This tells FastAPI exactly what shape of data to expect from the user
class ChatRequest(BaseModel):
    query: str

# This tells FastAPI what shape of data to send back
class ChatResponse(BaseModel):
    answer: str
    needs_action: str
    sources: list
    displayed_source: str

# 3. Define the n8n Webhook URL (Replace this with your actual n8n test URL later)
N8N_WEBHOOK_URL = "https://muhammed9935.app.n8n.cloud/webhook/create-ticket"

# 4. Create the API Endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        print(f"Received query: {request.query}")
        
        # Pass the query to your brain
        rag_result = rag_engine.run_rag_pipeline(request.query)
        
        # --- THE AUTOMATION TRIGGER ---
        # If the RAG says this is a complaint, fire off a request to n8n!
        # --- THE AUTOMATION TRIGGER ---
        if rag_result["needs_action"] == "YES":
            print("Action required! Sending data to n8n to create a ticket...")
            try:
                # Fire the request to n8n
                n8n_response = requests.post(N8N_WEBHOOK_URL, json={"customer_issue": request.query}, timeout=5)
                
                # Check if n8n successfully caught it (200 OK)
                if n8n_response.status_code == 200:
                    # Print the professional Arabic confirmation straight to the terminal
                    print("✅ n8n Confirmation Received: تم استلام التذكرة بنجاح في النظام")
                else:
                    print(f"⚠️ n8n returned an error: {n8n_response.status_code}")
                    
            except Exception as e:
                print(f"Warning: Failed to reach n8n webhook: {e}")
        # ------------------------------
        # ------------------------------

        # Return the final formatted response back to Streamlit
        return ChatResponse(
            answer=rag_result["answer"],
            needs_action=rag_result["needs_action"],
            sources=rag_result["sources"],
            displayed_source=rag_result["displayed_source"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Simple health check endpoint just to make sure the server is alive
@app.get("/")
def read_root():
    return {"status": "NileTel API is running!"}