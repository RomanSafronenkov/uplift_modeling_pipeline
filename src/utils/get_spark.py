import os
import sys
from pyspark.sql import SparkSession
from pyspark import SparkConf


def get_conf():
    _CONF = SparkConf().setAll([
        ('spark.ui.enabled', 'true')
    ])
    return _CONF

_CONF = get_conf()

def get_spark(conf=_CONF, app_name='spark'):
    spark = (
        SparkSession
        .builder
        .appName(app_name)
        .config(conf=conf)
        .getOrCreate()
    )
    return spark
