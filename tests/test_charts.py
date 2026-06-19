import numpy as np

from src.charts import response_dataframe


def test_columns_and_inband_filtering():
    freqs = np.array([0.0, 10.0, 100.0, 1000.0, 30000.0])  # 0,10 below band; 30000 above
    s = {"a": np.array([1., 2., 3., 4., 5.]), "b": np.array([10., 20., 30., 40., 50.])}
    df = response_dataframe(freqs, s)
    assert list(df.columns) == ["freq_hz", "magnitude_db", "series"]
    assert len(df) == 4  # in-band freqs {100,1000} x 2 series
    assert set(df["series"]) == {"a", "b"}
    assert sorted(df["freq_hz"].unique().tolist()) == [100.0, 1000.0]


def test_values_preserved_in_band():
    freqs = np.array([100.0, 1000.0, 10000.0])
    df = response_dataframe(freqs, {"x": np.array([1.0, 2.0, 3.0])})
    assert df["magnitude_db"].tolist() == [1.0, 2.0, 3.0]


def test_single_series_row_count():
    freqs = np.array([20.0, 200.0, 2000.0, 20000.0])
    df = response_dataframe(freqs, {"only": np.zeros(4)})
    assert len(df) == 4
    assert (df["series"] == "only").all()
