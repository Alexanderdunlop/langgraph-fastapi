from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, TypedDict, Annotated, Union
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
from pydantic import BaseModel

load_dotenv()

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

# Define the state type
class State(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    current_step: int
    max_steps: int

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set")

# Initialize the LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

# Define the agent function
def agent(state: State) -> dict:
    # Get the messages from the state
    messages = state["messages"]
    
    # Generate response from the LLM
    response = llm.invoke(messages)
    
    # Add the response to messages
    messages.append(response)
    
    # Update the state
    state["messages"] = messages
    state["current_step"] += 1
    
    # Decide whether to continue or end
    if state["current_step"] >= state["max_steps"]:
        return {"state": state, "next": END}
    return {"state": state, "next": "agent"}

# Create the graph
workflow = StateGraph(State)

workflow.add_node("agent", agent)

workflow.set_entry_point("agent")
workflow.add_edge("agent", END)
graph = workflow.compile()

class ChatRequest(BaseModel):
    message: str
    max_steps: int = 3

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Initialize the state
        initial_state = State(
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