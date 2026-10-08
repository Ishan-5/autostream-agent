# AutoStream AI Agent
### Social-to-Lead Agentic Workflow

## Quick Start

git clone https://github.com/Ishan-5/autostream-agent.git
cd autostream-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
python agent.py
 
## Architecture (~200 words)
 
This agent is built using LangGraph, a graph-based state machine framework.
LangGraph was chosen over AutoGen because it provides explicit, deterministic
control over conversation flow. AutoGen's multi-agent model would be overkill
for a single-agent lead-capture workflow.
 
State Management: The AgentState TypedDict persists across all 5-6 turns,
holding: message history, classified intent, lead fields (name, email, platform),
and boolean flags (collecting_lead, lead_captured).
 
Graph Flow:
  User -> detect_intent -> [route] -> collect_lead_info -> generate_response -> END
                                   -> generate_response -> END

RAG Pipeline: The knowledge base (JSON) is loaded at startup and injected
into the system prompt. For larger datasets, vector search (ChromaDB) would
replace this approach.
 
Tool Execution: mock_lead_capture() only fires when all 3 fields are confirmed.
 
## WhatsApp Integration
 
To deploy on WhatsApp using the WhatsApp Business Cloud API:
 
1. Create a Meta Developer App, enable WhatsApp Business product.
2. Build a FastAPI webhook endpoint that receives POST requests from WhatsApp.
3. For each incoming message, load or create the user's session state,
   call agent.invoke(state), extract the AI reply, and send it back via
   the WhatsApp Graph API (POST to /PHONE_ID/messages).
4. Register your webhook URL in Meta Developer Dashboard.
5. Use Redis (not in-memory dict) for production session storage.
