import logging
import mlflow
import os
import pickle as pkl
import xgboost as xgb

from croniter import croniter_range
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from prefect import flow, task

from electricity_demand_fc.data_retrieval import get_raster_points_inside_nl, get_X, get_y

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

logger = logging.getLogger("mlflow")
logger.setLevel(logging.INFO)


def parse_dts(dts: datetime | list[datetime] | str | list[str] | None = None) -> list[datetime]:
    if isinstance(dts, datetime):
        return [dts]
    if isinstance(dts, str):
        return [datetime.fromisoformat(dts)]
    elif isinstance(dts, list):
        return [datetime.fromisoformat(dt) if isinstance(dt, str) else dt for dt in dts]
    else:
        return [datetime.now()]


@task
def fit(X, y, eval_in_sample):
    model = xgb.XGBRegressor(objective="reg:squarederror")
    model.fit(X, y)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "model.pkl", "wb") as model_file:
        pkl.dump(model, model_file)

    if eval_in_sample:
        y_pred = model.predict(X)

        mse = mean_squared_error(y, y_pred, squared=False)
        rmse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R2 Score: {r2}")

        mlflow.log_metric("Mean Squared Error", mse)
        mlflow.log_metric("Root Mean Squared Error", rmse)
        mlflow.log_metric("Mean Absolute Error", mae)
        mlflow.log_metric("R2 Score", r2)


@task
def predict(X, y_test=None):
    with open("models/model.pkl", "rb") as model_file:
        model = pkl.load(model_file)
    y_pred = model.predict(X)

    if y_test is not None:
        mse = mean_squared_error(y_test, y_pred, squared=False)
        rmse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R2 Score: {r2}")

        mlflow.log_metric("Mean Squared Error", mse)
        mlflow.log_metric("Root Mean Squared Error", rmse)
        mlflow.log_metric("Mean Absolute Error", mae)
        mlflow.log_metric("R2 Score", r2)

    return y_pred


@task
def prepare_data(
    at_dts: list[datetime], fit_window: timedelta | None, predict_window: timedelta | None, eval_out_of_sample: bool
):
    if fit_window and predict_window:
        data_start_dt = at_dts[0] + fit_window
        data_end_dt = at_dts[-1] + predict_window
    elif not fit_window and predict_window:
        data_start_dt = at_dts[0]
        data_end_dt = at_dts[-1] + predict_window
    # When only wanting to fit, make sure only one date is given.
    # It does not make sense to only fit multiple times with no prediction inbetween
    elif fit_window and len(at_dts) == 1:
        data_start_dt = at_dts[0] + fit_window
        data_end_dt = at_dts[0]
    else:
        raise

    points = get_raster_points_inside_nl(raster_size=0.5)
    X = get_X(points, data_start_dt, data_end_dt)

    if fit_window or eval_out_of_sample:
        y = get_y(data_start_dt, data_end_dt)
    else:
        y = None

    return X, y


@flow(name="run")
def run(
    at_dt: datetime | list[datetime] | str | list[str] | None = None,
    start_dt: datetime | str | None = None,
    end_dt: datetime | str | None = None,
    cron_schedule: str | None = None,
    fit_window: timedelta | dict | None = None,
    predict_window: timedelta | dict | None = None,
    eval_in_sample: bool = False,
    eval_out_of_sample: bool = False,
):
    if start_dt and end_dt and cron_schedule:
        at_dts = croniter_range(start_dt, end_dt, cron_schedule)
    else:
        at_dts = parse_dts(at_dt)
    at_dts = sorted(at_dts)

    if isinstance(fit_window, dict):
        fit_window = timedelta(**fit_window)
    if isinstance(fit_window, dict):
        predict_window = timedelta(**predict_window)

    eval_in_sample = eval_in_sample
    eval_out_of_sample = eval_out_of_sample
    X, y = prepare_data(at_dts, fit_window, predict_window, eval_out_of_sample)

    for dt in at_dts:
        if fit_window:
            X_train = X[dt + fit_window : dt]
            y_train = y[dt + fit_window : dt]
            fit(X_train, y_train, eval_in_sample)
        if predict_window:
            X_test = X[dt : dt + predict_window]
            if eval_out_of_sample:
                y_test = y[dt : dt + predict_window]
            else:
                y_test = None
            predict(X_test, y_test)


class Run:
    def __init__(
        self,
        at_dt: datetime | list[datetime] | str | list[str] | None = None,
        start_dt: datetime | str | None = None,
        end_dt: datetime | str | None = None,
        cron_schedule: str | None = None,
        fit_window: timedelta | dict | None = None,
        predict_window: timedelta | dict | None = None,
        eval_in_sample: bool = False,
        eval_out_of_sample: bool = False,
    ) -> None:
        if start_dt and end_dt and cron_schedule:
            at_dts = croniter_range(start_dt, end_dt, cron_schedule)
        else:
            at_dts = self.parse_dts(at_dt)
        self.at_dts = sorted(at_dts)

        self.fit = False
        self.predict = False
        if isinstance(fit_window, dict):
            fit_window = timedelta(**fit_window)
        if isinstance(fit_window, dict):
            predict_window = timedelta(**predict_window)
        if fit_window:
            self.fit_window = fit_window
            self.fit = True
        if predict_window:
            self.predict_window = predict_window
            self.predict = True

        self.eval_in_sample = eval_in_sample
        self.eval_out_of_sample = eval_out_of_sample
        self.prepare_data()

        for dt in self.at_dts:
            self.run(dt)

    def run(self, dt):
        if self.fit:
            X_train = self.X[dt + self.fit_window : dt]
            y_train = self.y[dt + self.fit_window : dt]
            fit(X_train, y_train, self.eval_in_sample)
        if self.predict:
            X_test = self.X[dt : dt + self.predict_window]
            if self.eval_out_of_sample:
                y_test = self.y[dt : dt + self.predict_window]
            else:
                y_test = None
            predict(X_test, y_test)

    def prepare_data(self):
        if self.fit and self.predict:
            self.data_start_dt = self.at_dts[0] + self.fit_window
            self.data_end_dt = self.at_dts[-1] + self.predict_window
        elif not self.fit and self.predict:
            self.data_start_dt = self.at_dts[0]
            self.data_end_dt = self.at_dts[-1] + self.predict_window
        # When only wanting to fit, make sure only one date is given.
        # It does not make sense to only fit multiple times with no prediction inbetween
        elif self.fit and len(self.at_dts) == 1:
            self.data_start_dt = self.at_dts[0] + self.fit_window
            self.data_end_dt = self.at_dts[0]
        else:
            raise

        self.points = get_raster_points_inside_nl(raster_size=0.5)
        self.X = get_X(self.points, self.data_start_dt, self.data_end_dt)

        if self.fit or self.eval_out_of_sample:
            self.y = get_y(self.data_start_dt, self.data_end_dt)
        else:
            self.y = None

    def parse_dts(self, dts: datetime | list[datetime] | str | list[str] | None = None) -> list[datetime]:
        if isinstance(dts, datetime):
            return [dts]
        if isinstance(dts, str):
            return [datetime.fromisoformat(dts)]
        elif isinstance(dts, list):
            return [datetime.fromisoformat(dt) if isinstance(dt, str) else dt for dt in dts]
        else:
            return [datetime.now()]


if __name__ == "__main__":
    at_dt = datetime(2023, 8, 1)
    fit_window = timedelta(days=-1)
    predict_window = timedelta(days=1)
    run(at_dt=at_dt, fit_window=fit_window, predict_window=predict_window, eval_in_sample=True, eval_out_of_sample=True)
