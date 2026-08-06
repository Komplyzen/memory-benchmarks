import json

import pytest

from conductor import db, launcher, record, score
from conductor.cli import build_parser


def _isolated_conductor(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONDUCTOR_DB", str(tmp_path / "evals.db"))
    monkeypatch.setenv("CONDUCTOR_DATA_ROOT", str(tmp_path / "state"))
    db._initialized = False


def test_cli_exposes_the_evaluation_workflow():
    parser = build_parser()

    for argv in (
        ["start", "longmemeval", "--skip-preflight"],
        ["status", "run-id"],
        ["diff", "baseline", "candidate"],
        ["score", "run-id", "--scorer", "scorer.py"],
        ["verify-store", "store-id"],
    ):
        assert parser.parse_args(argv).command == argv[0]


def test_origin_captures_dataset_and_redacts_secrets(tmp_path, monkeypatch):
    _isolated_conductor(tmp_path, monkeypatch)
    dataset = tmp_path / "customer-sample.jsonl"
    dataset.write_text('{"question":"where?"}\n', encoding="utf-8")

    origin = record.capture_origin(
        benchmark="custom_eval",
        argv=["python", "-m", "benchmarks.custom_eval.run"],
        config={"dataset_path": str(dataset), "backend": "cloud"},
        env_overrides={"MEM0_API_KEY": "secret-value", "EVAL_MODE": "retrieval"},
        dataset_path=str(dataset),
    )

    assert origin["dataset"]["sha256"]
    assert origin["dataset"]["path"] == str(dataset.resolve())
    assert origin["env_overrides"]["MEM0_API_KEY"] == "secret…"
    assert origin["env_overrides"]["EVAL_MODE"] == "retrieval"


def test_arbitrary_scorer_records_mode_and_validity(tmp_path, monkeypatch):
    _isolated_conductor(tmp_path, monkeypatch)
    db.insert_run("custom-0001", "longmemeval", "customer-smoke", {}, {})
    db.insert_origin(
        "custom-0001",
        "longmemeval",
        None,
        "custom scorer smoke",
        {"benchmark": "longmemeval", "dataset": None},
    )
    scorer = tmp_path / "scorer.py"
    scorer.write_text(
        "import json, sys\n"
        "context = json.load(open(sys.argv[1]))\n"
        "print(json.dumps({\n"
        "  'mode': 'customer_retrieval',\n"
        "  'metrics': {'coverage': 0.75},\n"
        "  'validity': 'Directional within one fixed store.',\n"
        "  'headline': 0.75,\n"
        "  'headline_label': 'coverage',\n"
        "}))\n",
        encoding="utf-8",
    )

    verdict = score.score_run("custom-0001", str(scorer))
    recorded = json.loads(db.get_origin("custom-0001")["metrics_json"])

    assert verdict["mode"] == "customer_retrieval"
    assert recorded["derived_modes"]["customer_retrieval"]["metrics"]["coverage"] == 0.75
    assert "fixed store" in recorded["derived_modes"]["customer_retrieval"]["validity"]


def test_unknown_benchmark_fails_before_launch():
    with pytest.raises(ValueError, match="unknown benchmark"):
        launcher.start(
            benchmark="customer_unregistered",
            config={},
            skip_preflight=True,
        )
