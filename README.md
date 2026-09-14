# Employee Performance Dashboard

A Flask dashboard exploring employee/team event counts, notes, charts, and a serialized classification model. A separate FastHTML component implementation from the course is also retained.

Run from the root. The Flask entry point uses the root `employee_events.db`; the separate course package uses a different database path.

## Source guide

Paths below are relative to the implementation directory:

| Path | Purpose |
| --- | --- |
| `main.py` | Flask routes and inline page templates |
| `dashboard.py` | Separate FastHTML implementation |
| `base_components/`, `combined_components/` | UI components |
| `assets/model.pkl` | Serialized model; provenance needs documentation |
| `employee_events/` | Database/query package |
| `test_employee_events.py` | Database/table existence checks |
| `build_project.py` | Writes database/model assets; review before running |

## Local setup

```bash
git clone https://github.com/iimaha-AI/PerformanceDashboard-for-employee-data-analysis.git
cd PerformanceDashboard-for-employee-data-analysis
python -m venv .venv
```

Activate the environment, then:

```bash
python -m pip install -r requirements.txt
python main.py
```

Visit `http://127.0.0.1:5000`. Flask debug mode is enabled for local development. This procedure is not a claim that all routes have been validated.

## Validation and known issues

`python -m pytest test_employee_events.py` runs the existing checks, which do not cover web routes, model compatibility, or prediction quality.

The database under employee_events/ could not be read as SQLite, while the root database was readable. setup.py refers to missing employee_events/requirements.txt. The two database paths must not be treated as interchangeable.

The query package expects `employee_id`/`team_id` in entity tables, while the Flask database uses `id` and `name`. Choose and test a coherent schema before restructuring. Do not run the data-building script merely to hide these differences.

## Portfolio development

Add a verified demo, meaningful screenshot filenames, route/inference tests, and a reproducible training report. Confirm data provenance before claiming it is synthetic. Predictions remain a demonstration, not a validated real-world employment assessment.

Preserve the existing [license](LICENSE.txt). See [review notes](docs/REVIEW.md).
