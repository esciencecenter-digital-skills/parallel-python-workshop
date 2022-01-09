# parallel-python-workshop
![Python application](https://github.com/jhidding/parallel-python-workshop/workflows/Python%20application/badge.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/escience-academy/parallel-python-workshop/HEAD)

Environment for the Parallel Python workshop. Lesson material can be found on the [Software Carpentry Incubator](https://carpentries-incubator.github.io/lesson-parallel-python/)

If the tests pass, your setup is good for the workshop.

### For conda users

```bash
conda env create -f environment.yml
conda activate parallel-python
pytest
```

### Or use Poetry

```bash
pip install --user poetry
poetry install
poetry run pytest
```

