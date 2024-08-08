/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cudf/io/parquet.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

#include <ctime>

// RMM memory resource creation utilities
inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }
inline auto make_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
    make_cuda(), rmm::percent_of_free_device_memory(50));
}
inline auto make_managed() { return std::make_shared<rmm::mr::managed_memory_resource>(); }
inline auto make_managed_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
    make_managed(), rmm::percent_of_free_device_memory(50));
}

/**
 * @brief Create an RMM memory resource based on the type
 *
 * @param rmm_type Type of memory resource to create
 */
inline std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(
  std::string const& rmm_type)
{
  if (rmm_type == "cuda") return make_cuda();
  if (rmm_type == "pool") return make_pool();
  if (rmm_type == "managed") return make_managed();
  if (rmm_type == "managed_pool") return make_managed_pool();
  CUDF_FAIL("Unknown rmm_type parameter: " + rmm_type +
            "\nExpecting: cuda, pool, managed, or managed_pool");
}

/**
 * @brief Log the peak memory usage of the GPU
 */
class memory_stats_logger {
 public:
  memory_stats_logger()
    : existing_mr(rmm::mr::get_current_device_resource()),
      statistics_mr(rmm::mr::make_statistics_adaptor(existing_mr))
  {
    rmm::mr::set_current_device_resource(&statistics_mr);
  }

  ~memory_stats_logger() { rmm::mr::set_current_device_resource(existing_mr); }

  [[nodiscard]] size_t peak_memory_usage() const noexcept
  {
    return statistics_mr.get_bytes_counter().peak;
  }

 private:
  rmm::mr::device_memory_resource* existing_mr;
  rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource> statistics_mr;
};

/**
 * @brief Write a cudf::table to a parquet file
 *
 * @param table The cudf::table to write
 * @param path The path to write the parquet file to
 * @param col_names The names of the columns in the table
 */
void write_parquet(std::unique_ptr<cudf::table> table,
                   std::string const& path,
                   std::vector<std::string> const& col_names)
{
  CUDF_FUNC_RANGE();
  std::cout << __func__ << " : " << path << std::endl;
  cudf::io::table_metadata metadata;
  std::vector<cudf::io::column_name_info> col_name_infos;
  for (auto& col_name : col_names) {
    col_name_infos.push_back(cudf::io::column_name_info(col_name));
  }
  metadata.schema_info            = col_name_infos;
  auto const table_input_metadata = cudf::io::table_input_metadata{metadata};
  auto builder = cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(path));
  builder.metadata(table_input_metadata);
  auto const options = builder.build();
  cudf::io::parquet_chunked_writer(options).write(table->view());
}

/**
 * @brief Generate the `std::tm` structure from year, month, and day
 *
 * @param year The year
 * @param month The month
 * @param day The day
 */
std::tm make_tm(int year, int month, int day)
{
  std::tm tm{};
  tm.tm_year = year - 1900;
  tm.tm_mon  = month - 1;
  tm.tm_mday = day;
  return tm;
}

/**
 * @brief Calculate the number of days since the UNIX epoch
 *
 * @param year The year
 * @param month The month
 * @param day The day
 */
int32_t days_since_epoch(int year, int month, int day)
{
  std::tm tm             = make_tm(year, month, day);
  std::tm epoch          = make_tm(1970, 1, 1);
  std::time_t time       = std::mktime(&tm);
  std::time_t epoch_time = std::mktime(&epoch);
  double diff            = std::difftime(time, epoch_time) / (60 * 60 * 24);
  return static_cast<int32_t>(diff);
}
