"""
Canada AI Strategy Chatbot — Streamlit UI
Wraps the Part 5 CrewAI multi-agent pipeline with validation.

Usage:
  1. pip install streamlit crewai crewai-tools langchain langchain_community langchain_openai
             chromadb pypdf docx2txt tiktoken openai sentence_transformers duckduckgo_search pydantic
  2. Place your OpenAI API key in one of:
       - An environment variable  OPENAI_API_KEY
       - A file called  mie1624_api_key.txt  next to this script
  3. streamlit run app.py
"""

import os
import re
import streamlit as st

# ── LangChain / vector-store imports ─────────────────────────────────────────
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# ── CrewAI imports ───────────────────────────────────────────────────────────
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from duckduckgo_search import DDGS

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 300
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.2
DEFAULT_MAX_RETRIES = 3
DOCS_FOLDER = "./my_documents/"
PERSIST_DIR = "./canada_ai_vectorstore"

# ─────────────────────────────────────────────────────────────────────────────
# API KEY
# ─────────────────────────────────────────────────────────────────────────────
def load_api_key() -> str:
    """Try Streamlit secrets, then env var, then local file."""
    try:
        key = st.secrets["OPENAI_API_KEY"]
        if key:
            return key
    except (KeyError, FileNotFoundError):
        pass
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    for path in ("mie1624_api_key.txt", "api_key.txt"):
        if os.path.isfile(path):
            with open(path) as f:
                return f.readline().strip()
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# PART A — Knowledge Pipeline  (cached so it only runs once)
# ─────────────────────────────────────────────────────────────────────────────
DOCUMENT_METADATA = {
    "Part1_Report.docx":              {"type": "analysis",       "pillar": "all",            "part": "1"},
    "Part2_Report.docx":              {"type": "strategy",       "pillar": "all",            "part": "2"},
    "Part3_Report.docx":              {"type": "implementation", "pillar": "all",            "part": "3"},
    "Part4_Narrative_Framework.docx": {"type": "narrative",      "pillar": "all",            "part": "4"},
    "hai_ai_index_report_2025.pdf":   {"type": "summary",        "scope": "general",         "part": "1"},
    "trust-in-ai-en-report.pdf":      {"type": "source",         "country": "cross_country", "part": "1"},
}


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"(Page \d+ of \d+)", "", text)
    return text.strip()


@st.cache_resource(show_spinner="Building vector store …")
def build_vectorstore():
    """Load documents, chunk, embed, and return a FAISS vectorstore."""
    all_docs = []
    for fname in os.listdir(DOCS_FOLDER):
        fpath = os.path.join(DOCS_FOLDER, fname)
        if fname.endswith(".pdf"):
            docs = PyPDFLoader(fpath).load()
        elif fname.endswith(".docx"):
            docs = Docx2txtLoader(fpath).load()
        else:
            continue
        for d in docs:
            d.metadata["source_file"] = fname
        all_docs.extend(docs)

    for doc in all_docs:
        fname = doc.metadata.get("source_file", "")
        if fname in DOCUMENT_METADATA:
            doc.metadata.update(DOCUMENT_METADATA[fname])

    for doc in all_docs:
        doc.page_content = _clean_text(doc.page_content)
    all_docs = [d for d in all_docs if len(d.page_content) > 100]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " "],
    )
    chunks = splitter.split_documents(all_docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["char_count"] = len(chunk.page_content)
        source = chunk.metadata.get("source_file", "unknown")
        part = chunk.metadata.get("part", "")
        chunk.metadata["citation_label"] = f"Part {part} — {source}" if part else source

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# PART B — Agent Architecture  (cached so agents are created once)
# ─────────────────────────────────────────────────────────────────────────────
class RagSearchTool(BaseTool):
    name: str = "Knowledge Base Search"
    description: str = (
        "Search the Canada AI strategy knowledge base for relevant facts, policies, "
        "and analysis. Use for any question about Canada's AI strategy, "
        "global AI comparisons, or policy recommendations. "
        "Pass a 'query' argument with the search question."
    )

    def _run(self, query: str) -> str:
        try:
            vs = build_vectorstore()
            docs = vs.similarity_search(query, k=5)
            results = []
            for d in docs:
                label = d.metadata.get("citation_label", d.metadata.get("source_file", "unknown"))
                results.append(f"[{label}]\n{d.page_content}")
            return "\n\n---\n\n".join(results) if results else "No relevant documents found."
        except Exception as e:
            return f"Search failed: {e}"


class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = (
        "Search the web for recent news about Canada AI policy, "
        "global AI competitiveness, or AI strategy updates after 2025. "
        "Use ONLY when the question refers to recent events not in the knowledge base. "
        "Pass a 'query' argument with the search question."
    )

    def _run(self, query: str) -> str:
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=3):
                    results.append(f"[{r['title']}]\n{r['body']}\nURL: {r['href']}")
            return "\n\n---\n\n".join(results) if results else "No results found."
        except Exception as e:
            return f"Search failed: {e}"


@st.cache_resource(show_spinner="Initialising agents …")
def build_agents():
    """Create the CrewAI agents, tools, and validator LLM."""
    api_key = load_api_key()
    os.environ["OPENAI_API_KEY"] = api_key

    llm = LLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    rag_tool = RagSearchTool()
    web_search_tool = WebSearchTool()

    researcher = Agent(
        role="AI Policy Researcher",
        goal="Retrieve accurate, cited facts from the Canada AI knowledge base.",
        backstory="Expert researcher who analyzed 79 AI strategy documents from 15 countries.",
        tools=[rag_tool, web_search_tool],
        llm=llm, verbose=False, max_iter=3,
    )
    analyst = Agent(
        role="Strategy Analyst",
        goal="Interpret research through the lens of Canada AI Strategy 2.0 and its three pillars.",
        backstory="Senior strategy consultant who designed Canada AI Strategy 1.0 and 2.0.",
        tools=[rag_tool],
        llm=llm, verbose=False, max_iter=3,
    )
    writer = Agent(
        role="Policy Communicator",
        goal="Synthesize research and analysis into a clear, well-structured response.",
        backstory="Expert at making complex AI policy accessible. Never fabricates statistics.",
        tools=[],
        llm=llm, verbose=False, max_iter=2,
    )

    validator_llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=api_key)

    # Lightweight replacement for deprecated ConversationSummaryBufferMemory.
    # Chat history is managed by st.session_state.messages.
    class SimpleMemory:
        def __init__(self):
            self.history = []
        def save_context(self, inputs, outputs):
            self.history.append({"input": inputs.get("input", ""), "output": outputs.get("output", "")})
    memory = SimpleMemory()

    return researcher, analyst, writer, validator_llm, memory


# ─────────────────────────────────────────────────────────────────────────────
# PART B+C — Run pipeline with validation
# ─────────────────────────────────────────────────────────────────────────────

# Maps tool names → human-readable status lines
_TOOL_LABELS = {
    "Knowledge Base Search": ("🔍", "searching the knowledge base"),
    "Web Search":            ("🌐", "searching the web for recent information"),
}

def _make_step_callback(agent_label: str, status_container):
    """
    Return a step_callback for one agent.

    CrewAI calls this after every reasoning step with an object that is either:
      • AgentAction  — the agent decided to call a tool  (.tool, .tool_input)
      • AgentFinish  — the agent is done              (.return_values)
    We tolerate unknown shapes with a broad fallback.
    """
    def callback(step_output):
        if status_container is None:
            return
        try:
            # ── Tool call ──────────────────────────────────────────────────
            if hasattr(step_output, "tool"):
                tool_name = str(step_output.tool)
                icon, action = _TOOL_LABELS.get(tool_name, ("⚙️", f"using {tool_name}"))
                status_container.info(f"{icon} **{agent_label}** is {action}…")

            # ── Agent finished its task ────────────────────────────────────
            elif hasattr(step_output, "return_values"):
                status_container.info(f"✅ **{agent_label}** finished — passing results on…")

            # ── Observation / tool result coming back ──────────────────────
            elif isinstance(step_output, tuple) and len(step_output) == 2:
                action, _ = step_output
                if hasattr(action, "tool"):
                    tool_name = str(action.tool)
                    icon, action_label = _TOOL_LABELS.get(tool_name, ("⚙️", f"using {tool_name}"))
                    status_container.info(f"{icon} **{agent_label}** is {action_label}…")

        except Exception:
            # Never let a UI callback crash the pipeline
            pass

    return callback


def run_crew(question: str, researcher, analyst, writer,
             status_container=None) -> str:
    """Execute the three-agent sequential pipeline with live status updates."""

    # ── Attach per-agent step callbacks ──────────────────────────────────────
    researcher.step_callback = _make_step_callback("🔬 AI Policy Researcher", status_container)
    analyst.step_callback    = _make_step_callback("📊 Strategy Analyst",     status_container)
    writer.step_callback     = _make_step_callback("✍️ Policy Communicator",  status_container)

    # ── Task-transition announcements ─────────────────────────────────────────
    _task_sequence = [
        "📊 **Strategy Analyst** is interpreting findings through the Strategy 2.0 lens…",
        "✍️ **Policy Communicator** is writing the final response…",
        "✅ All three agents done — running validation…",
    ]
    _task_counter = {"n": 0}

    def _task_callback(task_output):
        idx = _task_counter["n"]
        if status_container is not None and idx < len(_task_sequence):
            status_container.info(_task_sequence[idx])
        _task_counter["n"] += 1

    # ── Tasks ─────────────────────────────────────────────────────────────────
    research_task = Task(
        description=(
            f"The user asked: '{question}'\n\n"
            "Search the knowledge base and retrieve all relevant facts. "
            "Note the source of each piece of information. "
            "If the question is about events after 2025, also use web search."
        ),
        expected_output="A structured list of relevant facts with their sources.",
        agent=researcher,
    )
    analysis_task = Task(
        description=(
            f"Based on the research findings for: '{question}'\n\n"
            "Interpret the facts in the context of Canada AI Strategy 2.0. "
            "Identify which strategic pillar or gap is most relevant."
        ),
        expected_output="A strategic interpretation connecting facts to Canada's AI strategy.",
        agent=analyst,
        context=[research_task],
    )
    writing_task = Task(
        description=(
            f"Write the final response to: '{question}'\n\n"
            "Combine research and analysis into one clear answer. "
            "Lead with a direct answer, then evidence, then the key implication."
        ),
        expected_output="A clear response of 150-300 words with cited evidence.",
        agent=writer,
        context=[research_task, analysis_task],
    )

    # Announce the very first agent before kickoff
    if status_container is not None:
        status_container.info("🔬 **AI Policy Researcher** is starting — retrieving facts from the knowledge base…")

    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,
        verbose=False,
        task_callback=_task_callback,
    )
    return str(crew.kickoff())


def validate_response(response: str, query: str, validator_llm) -> dict:
    """Check response against 4 validation rules."""
    prompt = f"""
    Review this response to the query: "{query}"
    Response: {response}
    
    Check:

    RULE 1 — NO UNSUPPORTED STATISTICS
    Every specific number (rankings, dollar amounts, percentages, 
    company counts) must be grounded in the retrieved context.
    Flag any statistic that appears invented or unverifiable.
    
    RULE 2 — NO COUNTRY CONFUSION  
    Claims about specific countries (Canada, Israel, Singapore, UK, 
    India, Germany) must be accurate. Flag any claim that attributes 
    the wrong policy or metric to the wrong country.
    
    RULE 3 — QUESTION IS ACTUALLY ANSWERED
    The response must directly answer what the user asked.
    Flag if the response is vague, off-topic, or only partially answers.

    RULE 4 — ALL CLAIMS MUST BE EVIDENCE-BASED
    Every claim in the response must be traceable to the retrieved 
    knowledge base. Flag any claim that appears to come from outside 
    the provided context or cannot be verified against the source documents. 
    Try to use both the retrieved context and any web search results to verify claims. 
    If the websearch tool was used, claims based on web search must include an external reference 
    (e.g., a URL, citation, or known authoritative source) 
    so the user can verify it independently. Never present unverified 
    claims as established fact.
    
    Respond in this exact format:
    STATUS: APPROVED or REVISE
    ISSUE: (if REVISE, describe the specific problem in one sentence)
    FIX: (if REVISE, write the exact instruction to fix it)
    """
    raw = validator_llm.invoke(prompt).content.strip()
    lines = {
        line.split(":")[0].strip(): ":".join(line.split(":")[1:]).strip()
        for line in raw.split("\n") if ":" in line
    }
    return {
        "status": lines.get("STATUS", "APPROVED"),
        "issue": lines.get("ISSUE", ""),
        "fix": lines.get("FIX", ""),
    }


def ask_with_validation(question, researcher, analyst, writer, validator_llm, memory, status_container, max_retries=5):
    """Full pipeline: crew → validate → self-correct up to max_retries."""
    st.session_state.stop_requested = False
    # Live agent status is now shown inside run_crew via step/task callbacks
    response = run_crew(question, researcher, analyst, writer, status_container=status_container)

    for attempt in range(max_retries):
        # ── Check if user clicked Stop ───────────────────────────────────
        if st.session_state.get("stop_requested", False):
            status_container.warning("🛑 **Stopped by user** — returning current response.")
            st.session_state.stop_requested = False
            return str(response)

        memory.save_context({"input": question}, {"output": str(response)})
        status_container.info(f"✅ **Validating …** (attempt {attempt + 1}/{max_retries})")
        result = validate_response(str(response), question, validator_llm)

        if result["status"] == "APPROVED":
            status_container.success(f"✅ **Validated** on attempt {attempt + 1}")
            return str(response)

        status_container.warning(f"🔄 **Revising** (attempt {attempt + 1}): {result['issue']}")
        correction_prompt = (
            f"The previous response had this issue: {result['issue']}\n"
            f"Fix instruction: {result['fix']}\n\n"
            f"Original response:\n{response}\n\n"
            f"Write a corrected version."
        )
        response = validator_llm.invoke(correction_prompt).content

    status_container.warning("⚠️ Max retries reached — returning best available response.")
    st.session_state.stop_requested = False
    return str(response)


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Canada AI Strategy Advisor",
        page_icon="🍁",
        layout="centered",
    )

    st.title("🍁 Canada AI Strategy Advisor")
    st.caption(
        "Powered by a CrewAI multi-agent pipeline — "
        "Researcher → Analyst → Writer — with self-correcting validation."
    )

    # ── Load API key silently from secrets/env ─────────────────────────────
    api_key = load_api_key()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("**Architecture**")
        st.markdown(
            "1. **Researcher** — retrieves facts via RAG + web search\n"
            "2. **Analyst** — interprets through Strategy 2.0 lens\n"
            "3. **Writer** — synthesizes a clear answer\n"
            "4. **Validator** — checks 4 rules, auto-fixes if needed"
        )
        st.divider()
        st.markdown("**Validation retries**")
        max_retries = st.number_input(
            "Max validation retries",
            min_value=1,
            max_value=20,
            value=DEFAULT_MAX_RETRIES,
            step=1,
            help="How many times the validator may revise the response before returning.",
        )
        st.divider()
        st.markdown("**Sample questions**")
        for q in [
            "What is Canada's biggest AI weakness?",
            "How does Israel's commercialization model work?",
            "What are the three pillars of Canada AI Strategy 2.0?",
            "Is Canada a global leader in AI?",
        ]:
            if st.button(q, use_container_width=True):
                st.session_state["prefill"] = q

    # ── Guard: need API key ──────────────────────────────────────────────────
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("No API key found. Set OPENAI_API_KEY in Streamlit secrets or as an environment variable.")
        st.stop()

    # ── Guard: need documents ────────────────────────────────────────────────
    if not os.path.isdir(DOCS_FOLDER) or not os.listdir(DOCS_FOLDER):
        st.error(
            f"No documents found in `{DOCS_FOLDER}`. "
            "Place your PDF/DOCX files there and restart."
        )
        st.stop()

    # ── Initialise resources (cached) ────────────────────────────────────────
    build_vectorstore()
    researcher, analyst, writer, validator_llm, memory = build_agents()

    # ── Session-state defaults ─────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Handle prefilled question from sidebar ───────────────────────────────
    prefill = st.session_state.pop("prefill", None)
    user_input = st.chat_input("Ask about Canada's AI strategy …") or prefill

    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run agent pipeline
        with st.chat_message("assistant"):
            status = st.empty()
            stop_col = st.empty()
            stop_col.button("⏹ Stop generating", key="stop_btn",
                            on_click=lambda: st.session_state.update(stop_requested=True))
            with st.spinner("Thinking …"):
                answer = ask_with_validation(
                    user_input, researcher, analyst, writer,
                    validator_llm, memory, status,
                    max_retries=max_retries,
                )
            stop_col.empty()          # remove stop button once done
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()