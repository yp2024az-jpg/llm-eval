# IPL Text-to-SQL LLM Benchmark

This project is a benchmark of language models on the task of translating natural language questions into SQLite queries over an IPL cricket dataset (2021-2024). A SQLite database is constructed from the CSV files, each model is run on the questions in the golden dataset and the execution accuracy is recorded.

## Requirements

- Python 3.9+
- NVIDIA AI Endpoints API keys configured per model
- The match and delivery CSV files in `data/`
- A golden_dataset.csv file in the project root

Install the dependencies:

```bash
python -m pip install pandas python-dotenv langchain-core langchain-nvidia-ai-endpoints
```

```dotenv
m1_api_key=your_first_nvidia_api_key
m2_api_key=your_second_nvidia_api_key
```

Keep `.env` private. Do not commit your API key.

## Choose the models to evaluate

The models and their API key variable names are all defined in [`model_nvidia.py`](./model_nvidia.py). Each entry in the list has a display name, the NVIDIA model ID and the corresponding environment variable name:

```python
MODELS = [
    ("Model one", "provider/model-id", "m1_api_key"),
    ("Model two", "provider/another-model-id", "m2_api_key"),
]
```

Set each of the referenced environment variables in .env to the corresponding NVIDIA API key. You can use the same key for multiple models by simply giving it the same environment variable name in each of their entries. Use the model IDs that are available to your NVIDIA account (this benchmark doesn't support other providers such as OpenRouter without some code changes).

## Run the benchmark

Build or refresh the SQLite database from the CSVs in `data/`:

```bash
python db.py
```

Run the evaluation (the script filename contains a space):

```bash
python "main .py"
```

The eval reads the included golden_dataset.csv (20 questions) and schema.sql and then writes the per question results out to eval_results.csv. The golden_dataset_generator.py script has its own set of hard questions and validates those against the DB, it doesn't generate or overwrite the golden_dataset.csv.

To try a single model query instead:

```bash
python first_test.py
```

## Project files

- `db.py` builds `ipl_2021_2024.db` from `data/matches.csv` and `data/deliveries.csv`.
- `main .py` runs the model evaluation and saves `eval_results.csv`.
- `model_nvidia.py` lists model names and NVIDIA AI Endpoints model IDs to check.
- evaluator.py - compares the generated query results with the expected results.
- golden_dataset_generator.py - contains the hard benchmark questions and validates the SQL.
- `schema.sql` is the database schema included in the model prompt.
- `first_test.py`, a smoke test with one question and one model.