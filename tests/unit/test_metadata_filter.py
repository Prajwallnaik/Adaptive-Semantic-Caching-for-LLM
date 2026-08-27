"""
Unit tests for metadata filter construction.
"""

from app.core.metadata_filter import build_filter
from qdrant_client.http.models import Filter

class TestMetadataFilter:
    def test_no_filters(self):
        f = build_filter(exclude_expired=False)
        assert f is None

    def test_user_id_filter(self):
        f = build_filter(user_id="user123", exclude_expired=False)
        assert isinstance(f, Filter)
        assert len(f.must) == 1
        assert f.must[0].key == "user_id"
        assert f.must[0].match.value == "user123"

    def test_all_filters(self):
        f = build_filter(
            user_id="user123",
            model="test-model",
            context_version="v2",
            exclude_expired=True,
        )
        assert isinstance(f, Filter)
        assert len(f.must) == 4
        keys = {cond.key for cond in f.must}
        assert keys == {"user_id", "model", "context_version", "expires_at"}

    def test_ttl_only(self):
        f = build_filter(exclude_expired=True)
        assert isinstance(f, Filter)
        assert len(f.must) == 1
        assert f.must[0].key == "expires_at"
