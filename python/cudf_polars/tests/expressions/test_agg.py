# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.dsl import expr
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.callback import execute_with_cudf


# Note: quantile is tested separately (since it takes another argument)
@pytest.fixture(params=sorted(expr.Agg._SUPPORTED - {"quantile"}))
def agg(request):
    return request.param


@pytest.fixture(params=[pl.Int32, pl.Float32, pl.Int16])
def dtype(request):
    return request.param


@pytest.fixture(
    params=[
        False,
        pytest.param(True, marks=pytest.mark.xfail(reason="No handler for set_sorted")),
    ],
    ids=["unsorted", "sorted"],
)
def is_sorted(request):
    return request.param


@pytest.fixture
def df(dtype, with_nulls, is_sorted):
    values = [-10, 4, 5, 2, 3, 6, 8, 9, 4, 4, 5, 2, 3, 7, 3, 6, -10, -11]
    if with_nulls:
        values = [None if v % 5 == 0 else v for v in values]

    if is_sorted:
        values = sorted(values, key=lambda x: -1000 if x is None else x)

    df = pl.LazyFrame({"a": values}, schema={"a": dtype})
    if is_sorted:
        return df.set_sorted("a")
    return df


def test_agg(df, agg):
    expr = getattr(pl.col("a"), agg)()
    q = df.select(expr)

    # https://github.com/rapidsai/cudf/issues/15852
    check_dtypes = agg not in {"n_unique", "median"}
    if not check_dtypes and q.collect_schema()["a"] != pl.Float64:
        with pytest.raises(AssertionError):
            assert_gpu_result_equal(q)
    assert_gpu_result_equal(q, check_dtypes=check_dtypes, check_exact=False)


@pytest.mark.parametrize("q", [0.5, pl.lit(0.5)])
@pytest.mark.parametrize("interp", ["nearest", "higher", "lower", "midpoint", "linear"])
def test_quantile(df, q, interp):
    expr = pl.col("a").quantile(q, interp)
    q = df.select(expr)

    # https://github.com/rapidsai/cudf/issues/15852
    check_dtypes = q.collect_schema()["a"] == pl.Float64
    if not check_dtypes:
        with pytest.raises(AssertionError):
            assert_gpu_result_equal(q)
    assert_gpu_result_equal(q, check_dtypes=check_dtypes, check_exact=False)


def test_quantile_invalid_q(df):
    expr = pl.col("a").quantile(pl.col("a"))
    q = df.select(expr)
    with pytest.raises(pl.exceptions.ComputeError, match="cudf-polars only supports expressions that evaluate to a scalar as the quantile argument"):
        q.collect(post_opt_callback=execute_with_cudf)



@pytest.mark.parametrize(
    "propagate_nans",
    [pytest.param(False, marks=pytest.mark.xfail(reason="Need to mask nans")), True],
    ids=["mask_nans", "propagate_nans"],
)
@pytest.mark.parametrize("op", ["min", "max"])
def test_agg_float_with_nans(propagate_nans, op):
    df = pl.LazyFrame({"a": pl.Series([1, 2, float("nan")], dtype=pl.Float64())})
    op = getattr(pl.Expr, f"nan_{op}" if propagate_nans else op)
    q = df.select(op(pl.col("a")))

    assert_gpu_result_equal(q)
