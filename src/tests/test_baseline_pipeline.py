from news_sentinel.models.baseline import BaselineTrainingInput, train_baseline_model


def test_baseline_pipeline_trains_and_predicts() -> None:
    train_texts = [
        "stock market rally earnings",
        "team wins championship game",
        "government election policy",
        "new ai chip technology",
    ]
    train_labels = [2, 1, 0, 3]

    model = train_baseline_model(
        BaselineTrainingInput(train_texts=train_texts, train_labels=train_labels)
    )
    preds = model.predict(["ai technology company", "football game team"])
    assert len(preds) == 2
