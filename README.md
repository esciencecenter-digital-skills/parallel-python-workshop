# parallel-python-workshop
![Python application](https://github.com/jhidding/parallel-python-workshop/workflows/Python%20application/badge.svg)

Environment for the Parallel Python workshop. Lesson material can be found on the [Software Carpentry Incubator](https://carpentries-incubator.github.io/lesson-parallel-python/)

## For conda users

``` {.bash}
conda env create -f environment.yml
conda activate parallel-python
pytest
```

## Or
Create a virtual environment and activate it.

``` {.bash}
virtualenv parallel-python
parallel-python/bin/activate.sh
```

Or better, use some `virtualenvwrapper` script; then,

``` {.bash}
pip install -r requirements.txt
pytest
```
