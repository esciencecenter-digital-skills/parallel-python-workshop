# parallel-python-workshop
![Python application](https://github.com/jhidding/parallel-python-workshop/workflows/Python%20application/badge.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/escience-academy/parallel-python-workshop/HEAD)

Environment for the Parallel Python workshop. Lesson material can be found on the [Software Carpentry Incubator](https://carpentries-incubator.github.io/lesson-parallel-python/)

If the tests pass, your setup is good for the workshop.

### UV

UV is our recommended tool to manage the Python environment. Please follow the [UV install instructions](https://docs.astral.sh/uv/#installation) if you haven't already. Then, running from the directory where you cloned this repository, run the following commands:

```bash
uv sync
uv run pytest
```

### Conda

If you want to use Conda instead of UV, you can do the following:

```bash
conda env create -f environment.yml
conda activate parallel-python
pytest
```

