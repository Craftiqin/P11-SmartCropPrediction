from predict import main


def test_cli_runs_saved_pipeline_and_reports_distinct_scores(capsys):
    result = main(
        [
            "--temperature", "28",
            "--humidity", "65",
            "--soil-moisture", "40",
            "--light", "12000",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Selected model: KNN (tuned)" in output
    assert "Feature order: temperature, humidity, soil_moisture, light" in output
    assert "Predicted crop:" in output
    assert "Top KNN vote fractions (not calibrated probabilities):" in output
    assert "Separate profile-based suitability (not model probability):" in output
    assert "not field-validated advice" in output
