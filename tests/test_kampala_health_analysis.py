import json
from pathlib import Path

import numpy as np

from src.kampala_health_analysis import (
    KAMPALA_DIVISIONS,
    OUTCOME_COLUMNS,
    aggregate_time_panel,
    build_spatial_panel,
    load_population,
    load_weekly_panel,
)


def test_weekly_panel_has_unique_dates_and_preserves_missing_outcomes():
    weekly = load_weekly_panel()
    assert weekly.date.is_unique
    assert set(OUTCOME_COLUMNS).issubset(weekly.columns)
    # Missing DHIS2 reports must not be silently converted to zero.
    assert weekly[OUTCOME_COLUMNS].isna().any().any()


def test_monthly_outcomes_equal_weekly_complete_case_sums():
    weekly = load_weekly_panel()
    monthly = aggregate_time_panel(weekly, "MS")
    for outcome in OUTCOME_COLUMNS:
        expected = weekly.set_index("date")[outcome].resample("MS").sum(min_count=1)
        actual = monthly.set_index("date")[outcome]
        assert np.allclose(expected.fillna(-1), actual.fillna(-1))


def test_population_and_spatial_offsets_are_valid_and_unmatched_parishes_remain_unmatched():
    population = load_population()
    assert set(population.division) == set(KAMPALA_DIVISIONS)
    assert not population.duplicated(["division", "parish_key"]).any()
    panel, records, quality = build_spatial_panel(Path("data"), frequency="W-SUN")
    assert (panel.population_offset > 0).all()
    assert set(panel.division).issubset(KAMPALA_DIVISIONS)
    assert records.parish_matched.dtype == bool
    assert records.parish_matched.sum() < len(records)
    assert quality.loc[quality.metric.eq("records_with_same_day_division_pm25"), "value"].iloc[0] > 0


def test_notebooks_are_valid_and_use_shared_pipeline():
    for filename in ["analysis.ipynb", "analysis_month.ipynb", "analysis_year.ipynb"]:
        notebook = json.loads(Path(filename).read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "src.kampala_health_analysis" in source
        assert notebook["nbformat"] == 4
