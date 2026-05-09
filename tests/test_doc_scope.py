from fin_graph_rag.ingest.graph_store import GraphStore
from fin_graph_rag.models import DocumentRecord


def test_doc_hint_3m_matches_mmm_filename(tmp_path):
    db = tmp_path / "graph.sqlite"
    store = GraphStore(db)
    store.init_schema(reset=True)
    store.upsert_documents([
        DocumentRecord(doc_id="doc_mmm", file_name="MMM_2022_10K.pdf", path="/tmp/MMM_2022_10K.pdf", sha1="x", num_pages=1)
    ])
    try:
        assert store.find_doc_ids_by_name_hint(["3M"]) == ["doc_mmm"]
    finally:
        store.close()


def test_query_infers_3m_alias():
    store = GraphStore(":memory:")
    try:
        hints = store.infer_doc_hints_from_query("What risks could pressure gross margin for 3M?")
        assert "3m" in hints
        assert "mmm" in hints
    finally:
        store.close()

from fin_graph_rag.evaluation.financebench import get_doc_hints


def test_financebench_doc_metadata_builds_filename_hint():
    row = {
        "company": "3M",
        "doc_type": "10k",
        "doc_period": 2018,
        "doc_link": "https://investors.3m.com/financials/sec-filings/content/0001558370-19-000470/0001558370-19-000470.pdf",
    }
    hints = get_doc_hints(row)
    assert "3M_2018_10K" in hints
    assert "0001558370-19-000470.pdf" in hints


def test_structured_financebench_hint_matches_local_3m_pdf(tmp_path):
    db = tmp_path / "graph.sqlite"
    store = GraphStore(db)
    store.init_schema(reset=True)
    store.upsert_documents([
        DocumentRecord(doc_id="doc_3m_2018", file_name="3M_2018_10K.pdf", path="/tmp/3M_2018_10K.pdf", sha1="x", num_pages=1),
        DocumentRecord(doc_id="doc_3m_2022", file_name="3M_2022_10K.pdf", path="/tmp/3M_2022_10K.pdf", sha1="y", num_pages=1),
    ])
    try:
        hints = get_doc_hints({"company": "3M", "doc_type": "10k", "doc_period": 2018})
        assert store.find_doc_ids_by_name_hint(hints) == ["doc_3m_2018"]
    finally:
        store.close()
