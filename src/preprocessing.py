import logging
import re
from typing import List, Tuple

import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T

from src.config import AppConfig


_logger = logging.getLogger(__name__)


def find_dt_features(sdf: pyspark.sql.session.SparkSession) -> list:
    def find_dt(col):
        res = re.search(r'.*(_dt).*', col)
        if res is not None:
            result = res.group(0)
            if result not in ['report_dt', 'start_dt']:
                return result

    return list(filter(find_dt, sdf.columns))


def count_dt_diff(sdf: pyspark.sql.session.SparkSession, dt_features: List[str], date_col: str) -> pyspark.sql.session.SparkSession:
    """For datetime features calculate difference between feature and report_dt"""
    for col in dt_features:
        sdf = sdf.withColumn(col+'_diff', F.datediff(F.col(date_col), F.col(col))).drop(col)
    return sdf


@F.udf(returnType=T.IntegerType())
def count_ones(col):
    if isinstance(col, str):
        events = list(col)
        s = 0
        for e in events:
            if e.isdigit():
                s += int(e)
        return s


def count_payroll(sdf: pyspark.sql.session.SparkSession, bitmask_features: List[str]) -> pyspark.sql.session.SparkSession:
    """Count sum for bitmask columns"""
    for col in bitmask_features:
        sdf = sdf.withColumn(col+'_sum', count_ones(col)).drop(col)
    return sdf


def preprocess_sdf(sdf: pyspark.sql.session.SparkSession, config: AppConfig) -> Tuple[: pyspark.sql.session.SparkSession, list, list]:
    """
    Preprocess spark dataframe, find dt features, find bitmask features, fill nans, transform decimals
    """
    dt_features = find_dt_features(sdf)
    _logger.info(f'Num of datetime features: {len(dt_features)}')

    if dt_features:
        sdf = count_dt_diff(sdf, dt_features, config.dataset.date_col)

    bitmask_cols = list(filter(lambda x: 'bitmask' in x, sdf.columns))
    _logger.info(f"Num of bitmask features: {len(bitmask_cols)}")
    if bitmask_cols:
        sdf = count_payroll(sdf, bitmask_cols)

    sdf = sdf.fillna(config.feature_selection.num_na_val)
    sdf = sdf.fillna(config.feature_selection.string_na_val)

    decimal_columns = list(filter(lambda x: 'decimal' in x[1], sdf.dtypes))
    decimal_columns = [col[0] for col in decimal_columns]
    _logger.info(f"Num of decimal features: {len(decimal_columns)}")

    for col_name in decimal_columns:
        sdf = sdf.withColumn(col_name, F.col(col_name).cast('float'))

    _logger.info(f"Unique datatypes in dataset: {set(map(lambda x: x[1], sdf.dtypes))}")

    return sdf, dt_features, bitmask_cols
