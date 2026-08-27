"""
Unit tests for Reciprocal Rank Fusion (RRF) math.
"""

from app.core.hybrid_search import _reciprocal_rank_fusion, SearchResult
from dataclasses import dataclass, field

@dataclass
class MockPoint:
    id: str
    score: float
    payload: dict = field(default_factory=dict)

class TestRRF:
    def test_rrf_basic_fusion(self):
        dense = [
            MockPoint("A", 0.95, {"query_text": "q_a", "answer": "a_a"}),
            MockPoint("B", 0.90, {"query_text": "q_b", "answer": "a_b"}),
            MockPoint("C", 0.85, {"query_text": "q_c", "answer": "a_c"}),
        ]
        sparse = [
            MockPoint("B", 5.0, {"query_text": "q_b", "answer": "a_b"}),
            MockPoint("C", 3.0, {"query_text": "q_c", "answer": "a_c"}),
            MockPoint("D", 1.0, {"query_text": "q_d", "answer": "a_d"}),
        ]

        results = _reciprocal_rank_fusion(dense, sparse, k=60)

        # B: dense rank 2, sparse rank 1 -> 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
        assert results[0].point_id == "B"
        assert abs(results[0].score - (1/62 + 1/61)) < 1e-6

        # C: dense rank 3, sparse rank 2 -> 1/63 + 1/62 = 0.015873 + 0.016129 = 0.032002
        assert results[1].point_id == "C"
        
        # A: dense rank 1 -> 1/61 = 0.016393
        assert results[2].point_id == "A"

        # D: sparse rank 3 -> 1/63 = 0.015873
        assert results[3].point_id == "D"

    def test_empty_lists(self):
        results = _reciprocal_rank_fusion([], [])
        assert len(results) == 0

    def test_one_empty_list(self):
        dense = [MockPoint("A", 0.9, {"query_text": "q", "answer": "a"})]
        results = _reciprocal_rank_fusion(dense, [])
        assert len(results) == 1
        assert results[0].point_id == "A"
        assert abs(results[0].score - (1/61)) < 1e-6
