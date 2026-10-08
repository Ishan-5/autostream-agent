"""
AutoStream Conversational AI Agent
Built with LangGraph + Groq (LLaMA 3)
Social-to-Lead Agentic Workflow for ServiceHive / Inflx
"""

import os
import json
import re
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()


# ─────────────────────────────────────────────
# 1. KNOWLEDGE BASE (RAG)
# ─────────────────────────────────────────────

def load_knowledge_base(path: str = "knowledge_base.json") -> str:
    """Load the local knowledge base and return as formatted string."""
    with open(path, "r") as f:
        data = json.load(f)

    kb_text = f"Product: {data['product']}\n"
    kb_text += f"Description: {data['description']}\n\n"

    kb_text += "=== PRICING PLANS ===\n"
    for plan in data["plans"]:
        kb_text += f"\n{plan['name']} --- {plan['price']}\n"
        for feat in plan["features"]:
            kb_text += f"  - {feat}\n"

    kb_text += "\n=== COMPANY POLICIES ===\n"
    for policy in data["policies"]:
        kb_text += f"  - {policy}\n"

    kb_text += "\n=== FAQs ===\n"
    for faq in data["faqs"]:
        kb_text += f"Q: {faq['question']}\nA: {faq['answer']}\n\n"

    return kb_text


KNOWLEDGE_BASE = load_knowledge_base()


# ─────────────────────────────────────────────
# 2. TOOL: MOCK LEAD CAPTURE
# ─────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Simulates calling a backend CRM API to capture a lead.
    In production, replace this with a real HTTP POST to your CRM.
    """
    print("\n" + "="*50)
    print("🎯 LEAD CAPTURED SUCCESSFULLY!")
    print(f"   Name     : {name}")
    print(f"   Email    : {email}")
    print(f"   Platform : {platform}")
    print("="*50 + "\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"


# ─────────────────────────────────────────────
# 3. AGENT STATE
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The complete state that flows through every node of the LangGraph.

    Fields:
        messages        : Full conversation history (append-only via add_messages)
        intent          : Classified intent of the latest user message
        lead_name       : Collected lead name (None until provided)
        lead_email      : Collected lead email (None until provided)
        lead_platform   : Collected lead platform (None until provided)
        lead_captured   : True once mock_lead_capture() has been called
        collecting_lead : True while the agent is in lead-collection mode
    """
    messages: Annotated[List, add_messages]
    intent: Optional[str]
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    collecting_lead: bool


# ─────────────────────────────────────────────
# 4. LLM SETUP — GROQ
# ─────────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
    max_tokens=1024,
)


# ─────────────────────────────────────────────
# 5. SYSTEM PROMPT (RAG injected)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are Aria, a friendly and helpful sales assistant for AutoStream — a SaaS platform
that provides automated AI-powered video editing tools for content creators.

## YOUR KNOWLEDGE BASE (use ONLY this for product/pricing questions):
{KNOWLEDGE_BASE}

## YOUR RESPONSIBILITIES:
1. Greet users warmly and answer general questions.
2. Answer product and pricing questions ACCURATELY using ONLY the knowledge base above.
3. Identify when a user shows HIGH INTENT (wants to sign up, try, or purchase).
4. When high intent is detected, collect: Name, Email, Creator Platform (YouTube, Instagram, etc.)
   - Ask for ONE piece of information at a time.
   - Do NOT ask for all three at once.
5. Never make up features, prices, or policies not in the knowledge base.
6. Be concise, warm, and professional.

## INTENT CLASSIFICATION (internal — do not tell users):
- "greeting"   : Simple hello, hi, hey, how are you
- "inquiry"    : Questions about product, pricing, features, policies
- "high_intent": User wants to sign up, try, purchase, get started, or expresses strong interest
"""


# ─────────────────────────────────────────────
# 6. INTENT DETECTION NODE
# ─────────────────────────────────────────────

def detect_intent(state: AgentState) -> AgentState:
    """
    Uses the LLM to classify the user's latest message intent.
    Updates state["intent"] with one of: greeting | inquiry | high_intent
    """
    if not state["messages"]:
        return state

    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        return state

    user_text = last_message.content

    intent_prompt = f"""
Classify the following user message into EXACTLY one of these three categories:
- "greeting"     : A casual hello or social opener with no product interest
- "inquiry"      : Questions about the product, features, pricing, or policies
- "high_intent"  : The user wants to sign up, try the product, purchase, or shows strong buying interest

User message: "{user_text}"

Respond with ONLY the category word. Nothing else. No explanation.
"""

    response = llm.invoke([HumanMessage(content=intent_prompt)])
    raw = response.content.strip().lower()

    if "high" in raw or "sign" in raw or "buy" in raw or "purchase" in raw:
        intent = "high_intent"
    elif "inquiry" in raw or "question" in raw or "product" in raw:
        intent = "inquiry"
    else:
        intent = "greeting"

    #print(f"[DEBUG] Detected intent: {intent}")
    return {**state, "intent": intent}


# ─────────────────────────────────────────────
# 7. LEAD COLLECTION NODE
# ─────────────────────────────────────────────

def collect_lead_info(state: AgentState) -> AgentState:
    """
    Extracts Name, Email, and Platform from the user's latest message.
    Updates the respective state fields if found.
    """
    if not state["messages"]:
        return state

    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        return state

    user_text = last_message.content
    new_state = {**state}

    # Extract email with regex
    if not new_state.get("lead_email"):
        email_match = re.search(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
            user_text
        )
        if email_match:
            new_state["lead_email"] = email_match.group(0)
            print(f"[DEBUG] Extracted email: {new_state['lead_email']}")

    # Extract platform keywords
    if not new_state.get("lead_platform"):
        platforms = [
            "youtube", "instagram", "tiktok", "facebook",
            "twitter", "x", "linkedin", "twitch"
        ]
        for p in platforms:
            if p in user_text.lower():
                new_state["lead_platform"] = p.capitalize()
                print(f"[DEBUG] Extracted platform: {new_state['lead_platform']}")
                break

    # Extract name using LLM
    if not new_state.get("lead_name"):
        name_prompt = f"""
Extract the person's full name from this message if they are providing their name.
If no name is present, respond with "NONE".
Message: "{user_text}"
Respond with ONLY the name or "NONE". No explanation.
"""
        name_response = llm.invoke([HumanMessage(content=name_prompt)])
        extracted = name_response.content.strip()
        if extracted.upper() != "NONE" and len(extracted) > 1 and len(extracted) < 50:
            new_state["lead_name"] = extracted
            print(f"[DEBUG] Extracted name: {new_state['lead_name']}")

    return new_state


# ─────────────────────────────────────────────
# 8. RESPONSE GENERATION NODE
# ─────────────────────────────────────────────

def generate_response(state: AgentState) -> AgentState:
    """
    Core node: generates the agent's reply based on the current state.
    Handles greeting, inquiry, lead collection, and lead capture trigger.
    """
    new_state = {**state}

    # If lead already captured, just wrap up
    if state.get("lead_captured"):
        msg = AIMessage(content=(
            "I already have your details and our team will be in touch very soon! "
            "Is there anything else I can help you with today?"
        ))
        new_state["messages"] = state["messages"] + [msg]
        return new_state

    # Check if we have all 3 lead fields -> trigger tool
    if (
        state.get("collecting_lead")
        and state.get("lead_name")
        and state.get("lead_email")
        and state.get("lead_platform")
    ):
        # CALL THE MOCK LEAD CAPTURE TOOL
        mock_lead_capture(
            name=state["lead_name"],
            email=state["lead_email"],
            platform=state["lead_platform"],
        )

        confirmation = (
            f"Perfect! I've got everything I need. Here's a summary:\n\n"
            f"  Name     : {state['lead_name']}\n"
            f"  Email    : {state['lead_email']}\n"
            f"  Platform : {state['lead_platform']}\n\n"
            f"Our team will reach out to you at {state['lead_email']} within 24 hours "
            f"to set up your AutoStream Pro trial. Welcome aboard! 🚀"
        )
        msg = AIMessage(content=confirmation)
        new_state["messages"] = state["messages"] + [msg]
        new_state["lead_captured"] = True
        return new_state

    # Build context-aware prompt
    intent = state.get("intent", "inquiry")
    collecting = state.get("collecting_lead", False)

    context_note = ""
    if collecting:
        missing = []
        if not state.get("lead_name"):
            missing.append("full name")
        if not state.get("lead_email"):
            missing.append("email address")
        if not state.get("lead_platform"):
            missing.append("creator platform (e.g. YouTube, Instagram)")
        if missing:
            context_note = (
                f"\n\n[INTERNAL NOTE: You are collecting lead info. "
                f"Still need: {missing[0]}. "
                f"Ask for ONLY this one piece of information. Nothing else.]"
            )
    elif intent == "high_intent":
        context_note = (
            "\n\n[INTERNAL NOTE: The user has shown HIGH INTENT. "
            "Acknowledge their interest enthusiastically, then ask for their "
            "FULL NAME only. Do not ask for email or platform yet.]"
        )

    full_system = SYSTEM_PROMPT + context_note

    # Full conversation history sent to LLM
    lc_messages = [SystemMessage(content=full_system)] + list(state["messages"])
    response = llm.invoke(lc_messages)
    msg = AIMessage(content=response.content)
    new_state["messages"] = state["messages"] + [msg]

    # Set collecting_lead flag when high intent first detected
    if intent == "high_intent" and not collecting:
        new_state["collecting_lead"] = True

    return new_state


# ─────────────────────────────────────────────
# 9. ROUTING LOGIC
# ─────────────────────────────────────────────

def route_after_intent(state: AgentState) -> str:
    """
    Determines the next node after intent detection.
    If collecting lead info, route through collect_lead_info first.
    """
    if state.get("collecting_lead") and not state.get("lead_captured"):
        return "collect_lead_info"
    return "generate_response"


# ─────────────────────────────────────────────
# 10. BUILD THE LANGGRAPH
# ─────────────────────────────────────────────

def build_agent():
    """
    Constructs and compiles the LangGraph state machine.

    Graph flow:
        detect_intent -> [route] -> collect_lead_info -> generate_response -> END
                                 -> generate_response -> END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("collect_lead_info", collect_lead_info)
    graph.add_node("generate_response", generate_response)

    # Entry point
    graph.set_entry_point("detect_intent")

    # Conditional routing after intent detection
    graph.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {
            "collect_lead_info": "collect_lead_info",
            "generate_response": "generate_response",
        }
    )

    # After lead collection, always generate response
    graph.add_edge("collect_lead_info", "generate_response")

    # After response, end the turn
    graph.add_edge("generate_response", END)

    return graph.compile()


# ─────────────────────────────────────────────
# 11. MAIN CHAT LOOP
# ─────────────────────────────────────────────

def main():
    """Run the AutoStream agent in an interactive terminal loop."""
    print("\n" + "="*60)
    print("  🎬 AutoStream AI Assistant (powered by Groq + LLaMA 3)")
    print("  Type 'quit' or 'exit' to end the session")
    print("="*60 + "\n")

    agent = build_agent()

    # Initialize state
    state: AgentState = {
        "messages": [],
        "intent": None,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "collecting_lead": False,
    }

    print("Aria: Hi there! 👋 I'm Aria, your AutoStream assistant.")
    print("      How can I help you today?\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("\nAria: Thanks for chatting! Have a great day! 🎬\n")
            break

        # Add user message to state
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Run the graph for one turn
        result = agent.invoke(state)

        # Update state
        state = result

        # Print latest AI response
        last_ai_msg = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            None
        )
        if last_ai_msg:
            print(f"\nAria: {last_ai_msg.content}\n")


if __name__ == "__main__":
    main()