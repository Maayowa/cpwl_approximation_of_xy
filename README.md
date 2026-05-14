# cpwl_approximation_of_xy

## Setup

### With `pip`
```
python -m venv venv
source venv/bin/activate
pip install -e .
```

### With `uv`

```
uv venv
source .venv/bin/activate
uv add -r requirements.txt
uv pip install -e .
```

## Running TempRegPy model test
```
python scripts/script-run_tempregpy_sims.py
```


## Installing for use in *other* projects

This will *not* work within the repository and must be done in a separate project.


### With `pip`
```
pip install git+https://github.com/quentinplsrd/cpwl_approximation_of_xy
```

### With `uv`
```
uv add git+https://github.com/quentinplsrd/cpwl_approximation_of_xy
```