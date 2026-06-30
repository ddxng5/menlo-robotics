# Evaluation Submissions

Place local team Python submissions here:

```text
submissions/
  interim/
    team_01_solution.py
  final/
    team_01_solution.py
```

Each solution module should expose:

```python
async def run_agent(ctx, *, task, max_cycles=10_000, completion=None):
    ...
```

Run each team in an interactive session after applying the shared evaluation setup:

```text
custom menlo_runner.programs.evaluation_setup
complete submissions.interim.team_01_solution --level 0
```
