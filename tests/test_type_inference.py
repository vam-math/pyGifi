import pandas as pd

from pygifi.utils.type_inference import (
    coerce_dataframe_by_kinds,
    infer_column_kinds,
    prepare_dataframe_with_inference,
)


def test_infer_column_kinds_mixed_data():
    df = pd.DataFrame(
        {
            "city": ["A", "B", "A", "C"],
            "rating": [1, 2, 1, 3],
            "income": [1250.5, 1400.0, 1325.25, 1800.75],
            "score_text": ["1.1", "2.3", "3.7", "4.2"],
        }
    )

    inferred = infer_column_kinds(df)

    assert inferred["city"].kind == "categorical"
    assert inferred["rating"].kind == "categorical"
    assert inferred["income"].kind == "numerical"
    assert inferred["score_text"].kind == "numerical"


def test_prepare_dataframe_with_inference_noninteractive():
    df = pd.DataFrame(
        {
            "group": ["x", "y", "x", "z"],
            "value": [10.5, 11.5, 12.0, 13.25],
        }
    )

    prepared, resolved, _ = prepare_dataframe_with_inference(df, interactive=False)

    assert str(prepared["group"].dtype) == "category"
    assert pd.api.types.is_float_dtype(prepared["value"])
    assert resolved["group"] == "categorical"
    assert resolved["value"] == "numerical"


def test_coerce_dataframe_by_kinds_override():
    df = pd.DataFrame({"coded": [1, 2, 1, 3], "value": ["1.5", "2.0", "3.5", "4.0"]})

    coerced = coerce_dataframe_by_kinds(df, {"coded": "categorical", "value": "numerical"})

    assert str(coerced["coded"].dtype) == "category"
    assert pd.api.types.is_float_dtype(coerced["value"])
