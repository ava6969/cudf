# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp cimport bool
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.io.types cimport SourceInfo, TableWithMetadata
from cudf._lib.pylibcudf.libcudf.types cimport size_type
from cudf._lib.pylibcudf.types cimport DataType


cpdef TableWithMetadata read_orc(
    SourceInfo source_info,
    list columns = *,
    list stripes = *,
    size_type skip_rows = *,
    size_type num_rows = *,
    bool use_index = *,
    bool use_np_dtypes = *,
    DataType timestamp_type = *,
    list decimal128_columns = *
)
