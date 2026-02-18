from sklearn.feature_extraction.text import TfidfVectorizer

from news_sentinel.drift.checks import (
    DriftThresholds,
    class_prior_total_variation,
    summarize_drift,
    tfidf_centroid_cosine_distance,
)


def test_class_prior_total_variation_nonzero() -> None:
    ref = [0, 0, 1, 1]
    cur = [0, 2, 2, 2]
    out = class_prior_total_variation(ref, cur)
    assert out["total_variation_distance"] > 0.0


def test_embedding_shift_distance_in_range() -> None:
    ref_texts = ["stock market gains", "business earnings rise"]
    cur_texts = ["football team wins", "sports league final"]

    vec = TfidfVectorizer().fit(ref_texts + cur_texts)
    out = tfidf_centroid_cosine_distance(ref_texts, cur_texts, vec)

    assert 0.0 <= out["cosine_similarity"] <= 1.0
    assert 0.0 <= out["cosine_distance"] <= 1.0


def test_summarize_drift_warns_when_threshold_crossed() -> None:
    summary = summarize_drift(
        class_prior_tvd=0.2,
        embedding_distance=0.1,
        thresholds=DriftThresholds(class_prior_tvd_warn=0.15, embedding_shift_warn=0.2),
    )
    assert summary["status"] == "warn"
