# MLOps Zoomcamp Project - Electricity Demand Forecast

Forecasting electricity demand is essential for a reliable energy grid. Variations in demand can cause power disruptions and inefficiencies, leading to increased costs. With the growth of renewable energy, which can be unpredictable, precise demand forecasting has become increasingly important.

This project offers a comprehensive end-to-end solution designed for easy testing, deployment, and monitoring. This ensures we stay prepared for any unpredictable scenarios that may arise. The model predicts the demand for electricty using the Global Forecast System (GFS) weather data. The weather data is sourced via the [Herbie package](https://github.com/blaylockbk/Herbie) and the demand data is fetched from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) using the [entsoe-py](https://github.com/EnergieID/entsoe-py) package.

## Installation

### Running locally

Python version: 3.11
OS: Unix/Linux

Install system-dependency `libeccodes-dev` required by `cfgrib`, needed for downloading GFS data (for Debian/Ubuntu):

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

Install the git hook scripts

```bash
pre-commit install
```

Run pre-commit manually:

```bash
pre-commit run --all-files
```

### Running on the cloud

## MLFlow Server

You can access the MLFlow UI [here](http://ec2-3-75-231-58.eu-central-1.compute.amazonaws.com:5000/)

To set it up on AWS, I followed the steps in [MLOPS Zoomcamp 2.6 - Scenario 3](https://youtu.be/1ykg4YmbFVA?feature=shared&t=1165)

## Prefect Server

You can access the Prefect Server [here](http://ec2-3-75-231-58.eu-central-1.compute.amazonaws.com:4200/)

Steps I took to set it up on EC2:

Add two inbound rules with port 4200 and 4201 and choose source 0.0.0.0/0 to the security group (as shown in  [MLOPS Zoomcamp 2.6 - Scenario 3](https://youtu.be/1ykg4YmbFVA?feature=shared&t=1488))

```bash
pip install prefect
prefect server start --host 0.0.0.0
```
