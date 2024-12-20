from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from my_agent.agent import AgentState, graph

# Initialize FastAPI app
app = FastAPI(title="FastAPI App",
             description="A FastAPI application with LangGraph integration",
             version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    max_steps: int = 3

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Initialize the state
        initial_state = AgentState(
            messages=[HumanMessage(content=request.message)],
            current_step=0,
            max_steps=request.max_steps
        )
        
        # Run the chain
        result = graph.invoke(initial_state)
        
        # Extract the messages
        messages = result["messages"]
        responses = [msg.content for msg in messages if isinstance(msg, AIMessage)]
        
        return {"responses": responses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI with LangGraph!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 