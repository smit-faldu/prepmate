"""
app/vc/graph.py — LangGraph graph assembly and session management.

The graph is used exclusively as a SQLite-backed state store (get_state /
update_state). stream_vc_turn() in agent.py drives execution directly so we
control concurrency and streaming; the compiled graph's checkpointer is
reused purely for state persistence across turns.
"""

import sqlite3
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.vc.schemas import PitchState


def _noop_node(state: PitchState) -> dict:
    return {}


def build_vc_graph(db_path: str = "pitch_sessions.db"):
    conn   = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)

    graph = StateGraph(PitchState)
    graph.add_node("noop", _noop_node)
    graph.add_edge(START, "noop")
    graph.add_edge("noop", END)

    return graph.compile(checkpointer=memory)


# ── Singletons ────────────────────────────────────────────────────────────────

_vc_graph = None
_initialized_sessions: set[str] = set()


def get_vc_graph():
    global _vc_graph
    if _vc_graph is None:
        _vc_graph = build_vc_graph()
    return _vc_graph


def new_session() -> str:
    """Return a fresh unique session ID."""
    return str(uuid.uuid4())


def ensure_session_initialized(session_id: str) -> None:
    """Seed default state for a brand-new session (idempotent)."""
    if session_id in _initialized_sessions:
        return
    graph  = get_vc_graph()
    config = {"configurable": {"thread_id": session_id}}
    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.values:
        graph.update_state(
            config,
            {
                "messages":      [],
                "pitch_metrics": {},
                "stage":         "intro",
                "exchange_count": 0,
                "is_out":        False,
                "vc_response":   "",
                "pitch_ended":   False,
            },
        )
    _initialized_sessions.add(session_id)
