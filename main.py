from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from my_agent.agent import graph
from my_agent.utils.schemas import GraphConfig
from my_agent.utils.state import AgentState

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
    thread_id: str
    user_id: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Initialize the state
        initial_state = AgentState(
            messages=[HumanMessage(content=request.message)],
            core_memories=[],
            recall_memories=[]
        )

        config = GraphConfig(
            input=request.message,
            chat_history=[],
            context=[],
            thread_id=request.thread_id,
            user_id=request.user_id,
        )
        
        # Run the chain
        result = graph.invoke(initial_state, config)
        
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