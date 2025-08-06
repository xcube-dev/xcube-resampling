# About the `xcube-resampling` project

## Changelog

You can find the complete `xcube-resampling` changelog 
[here](https://github.com/xcube-dev/xcube-resampling/blob/main/CHANGES.md). 

## Reporting

If you have suggestions, ideas, feature requests, or if you have identified
a malfunction or error, then please 
[post an issue](https://github.com/xcube-dev/xcube-resampling/issues). 

## Contributions

The `xcube-resampling` project welcomes contributions of any form
as long as you respect our 
[code of conduct](https://github.com/xcube-dev/xcube-resampling/blob/main/CODE_OF_CONDUCT.md)
and follow our 
[contribution guide](https://github.com/xcube-dev/xcube-resampling/blob/main/CONTRIBUTING.md).

If you'd like to submit code or documentation changes, we ask you to provide a 
pull request (PR) 
[here](https://github.com/xcube-dev/xcube-resampling/pulls). 
For code and configuration changes, your PR must be linked to a 
corresponding issue. 

## Development

To install the `xcube-resampling` development environment into an existing Python 
environment, do

```bash
pip install .[dev,doc]
```

or create a new environment using `conda` or `mamba`

```bash
mamba env create 
```

### Testing and Coverage

`xcube-resampling` uses [pytest](https://docs.pytest.org/) for unit-level testing 
and code coverage analysis.

```bash
pytest tests/ --cov=xcube_resampling --cov-report html
```

### Code Style

The `xcube-resampling` source code is formatted and quality-controlled 
using [ruff](https://docs.astral.sh/ruff/):

```bash
ruff format
ruff check
```

### Documentation

The `xcube-resampling` documentation is built using the 
[mkdocs](https://www.mkdocs.org/) tool.

With repository root as current working directory:

```bash
pip install .[doc]

mkdocs build
mkdocs serve
mkdocs gh-deploy
```

## License

`xcube-resampling` is open source made available under the terms and conditions of the 
[MIT license](https://opensource.org/license/mit).
