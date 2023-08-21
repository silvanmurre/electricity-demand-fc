# MLOps Zoomcamp Project - Electricity Demand Forecast

The model predicts energy demand using Global Forecast System (GFS) weather data. Weather data is sourced via the [Herbie package](https://github.com/blaylockbk/Herbie), while demand data is fetched from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) using the [entsoe-py](https://github.com/EnergieID/entsoe-py) package.

## Installation

### Running locally

Python version: 3.11
OS: Unix/Linux

Install system-dependency `libeccodes-dev` required by `cfgrib` (for Debian/Ubuntu):

```bash
sudo apt install libeccodes-dev
```

Create virtual environment & activate:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -e .[dev]
```

### Running on the cloud
