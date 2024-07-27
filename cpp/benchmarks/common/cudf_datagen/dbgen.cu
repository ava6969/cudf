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
#include "schema.hpp"
#include "utils.hpp"
#include "vocab.hpp"

std::unique_ptr<cudf::column> l_calc_suppkey(cudf::column_view const& l_partkey,
                                             int64_t const& scale_factor,
                                             int64_t const& num_rows)
{
  // Generating the `s` col
  auto s_empty = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                           num_rows,
                                           cudf::mask_state::UNALLOCATED,
                                           cudf::get_default_stream());

  auto s =
    cudf::fill(s_empty->view(), 0, num_rows, cudf::numeric_scalar<int64_t>(10000 * scale_factor));

  // Generating the `i` col
  auto i = gen_rep_seq_col(4, num_rows);

  // Create a table view out of `l_partkey`, `s`, and `i`
  auto table = cudf::table_view({l_partkey, s->view(), i->view()});

  // Create the AST expression
  auto scalar_1  = cudf::numeric_scalar<int64_t>(1);
  auto scalar_4  = cudf::numeric_scalar<int64_t>(4);
  auto literal_1 = cudf::ast::literal(scalar_1);
  auto literal_4 = cudf::ast::literal(scalar_4);

  auto l_partkey_col_ref = cudf::ast::column_reference(0);
  auto s_col_ref         = cudf::ast::column_reference(1);
  auto i_col_ref         = cudf::ast::column_reference(2);

  // (int)(l_partkey - 1)/s
  auto expr_a = cudf::ast::operation(cudf::ast::ast_operator::SUB, l_partkey_col_ref, literal_1);
  auto expr_b = cudf::ast::operation(cudf::ast::ast_operator::DIV, expr_a, s_col_ref);
  auto expr_b_casted = cudf::ast::operation(cudf::ast::ast_operator::CAST_TO_INT64, expr_b);

  // s/4
  auto expr_c = cudf::ast::operation(cudf::ast::ast_operator::DIV, s_col_ref, literal_4);

  // (s/4 + (int)(l_partkey - 1)/s)
  auto expr_d = cudf::ast::operation(cudf::ast::ast_operator::ADD, expr_c, expr_b_casted);

  // (i * (s/4 + (int)(l_partkey - 1)/s))
  auto expr_e = cudf::ast::operation(cudf::ast::ast_operator::MUL, i_col_ref, expr_d);

  // (l_partkey + (i * (s/4 + (int)(l_partkey - 1)/s)))
  auto expr_f = cudf::ast::operation(cudf::ast::ast_operator::ADD, l_partkey_col_ref, expr_e);

  // (l_partkey + (i * (s/4 + (int)(l_partkey - 1)/s))) % s
  auto expr_g = cudf::ast::operation(cudf::ast::ast_operator::MOD, expr_f, s_col_ref);

  // (l_partkey + (i * (s/4 + (int)(l_partkey - 1)/s))) % s + 1
  auto final_expr = cudf::ast::operation(cudf::ast::ast_operator::ADD, expr_g, literal_1);

  // Execute the AST expression
  auto l_suppkey = cudf::compute_column(table, final_expr);
  return l_suppkey;
}

int64_t l_calc_cardinality(cudf::column_view const& o_orderkey_repeat_freqs)
{
  auto const sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto const l_num_rows_scalar =
    cudf::reduce(o_orderkey_repeat_freqs, *sum_agg, cudf::data_type{cudf::type_id::INT64});
  return reinterpret_cast<cudf::numeric_scalar<int64_t>*>(l_num_rows_scalar.get())->value();
}

/**
 * @brief Calculate the charge column
 *
 * @param tax The tax column
 * @param disc_price The discount price column
 * @param stream The CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
[[nodiscard]] std::unique_ptr<cudf::column> l_calc_charge(
  cudf::column_view const& extendedprice,
  cudf::column_view const& tax,
  cudf::column_view const& discount,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  auto const one = cudf::numeric_scalar<double>(1);
  auto const one_minus_discount =
    cudf::binary_operation(one, discount, cudf::binary_operator::SUB, discount.type(), stream, mr);
  auto const disc_price_type = cudf::data_type{cudf::type_id::FLOAT64};
  auto disc_price            = cudf::binary_operation(extendedprice,
                                           one_minus_discount->view(),
                                           cudf::binary_operator::MUL,
                                           disc_price_type,
                                           stream,
                                           mr);
  auto const one_plus_tax =
    cudf::binary_operation(one, tax, cudf::binary_operator::ADD, tax.type(), stream, mr);
  auto const charge_type = cudf::data_type{cudf::type_id::FLOAT64};
  auto charge            = cudf::binary_operation(
    disc_price->view(), one_plus_tax->view(), cudf::binary_operator::MUL, charge_type, stream, mr);
  return charge;
}

// NOTE: Incomplete table
void generate_lineitem_and_orders(int64_t scale_factor)
{
  cudf::size_type const num_rows = 1'500'000 * scale_factor;

  // Generate the `orders` table
  // Generate a primary key column for the orders table
  // which will not be written into the parquet file
  auto const o_pkey = gen_primary_key_col(0, num_rows);

  // Generate the `o_orderkey` column
  // NOTE: This column is not compliant with the specifications
  auto const o_orderkey_candidates = gen_primary_key_col(1, 4 * num_rows);
  auto const o_orderkey_unsorted   = cudf::sample(cudf::table_view({o_orderkey_candidates->view()}),
                                                num_rows,
                                                cudf::sample_with_replacement::FALSE);
  auto const o_orderkey =
    cudf::sort_by_key(o_orderkey_unsorted->view(),
                      cudf::table_view({o_orderkey_unsorted->view().column(0)}))
      ->get_column(0);

  // Generate the `o_custkey` column
  // NOTE: Currently, this column does not comply with the specs which
  // specifies that every value % 3 != 0
  auto const o_custkey = gen_rand_num_col<int64_t>(1, num_rows, num_rows);

  // Generate the `o_orderstatus` column

  // Generate the `o_orderdate` column
  // Uniformly distributed random dates between `1992-01-01` and `1998-08-02`
  auto const o_orderdate_year  = gen_rand_str_col_from_set(years, num_rows);
  auto const o_orderdate_month = gen_rand_str_col_from_set(months, num_rows);
  auto const o_orderdate_day   = gen_rand_str_col_from_set(days, num_rows);
  auto const o_orderdate_str   = cudf::strings::concatenate(
    cudf::table_view(
      {o_orderdate_year->view(), o_orderdate_month->view(), o_orderdate_day->view()}),
    cudf::string_scalar("-"));

  auto const o_orderdate_ts =
    cudf::strings::to_timestamps(o_orderdate_str->view(),
                                 cudf::data_type{cudf::type_id::TIMESTAMP_DAYS},
                                 std::string("%Y-%m-%d"));

  // Generate the `o_orderpriority` column
  auto const o_orderpriority = gen_rand_str_col_from_set(vocab_priorities, num_rows);

  // Generate the `o_clerk` column
  auto const clerk_repeat = gen_rep_str_col("Clerk#", num_rows);
  auto const random_c     = gen_rand_num_col<int64_t>(1, 1'000 * scale_factor, num_rows);
  auto const random_c_str = cudf::strings::from_integers(random_c->view());
  auto const random_c_str_padded =
    cudf::strings::pad(random_c_str->view(), 9, cudf::strings::side_type::LEFT, "0");
  auto const o_clerk = cudf::strings::concatenate(
    cudf::table_view({clerk_repeat->view(), random_c_str_padded->view()}));

  // Generate the `o_shippriority` column
  auto const empty = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                               num_rows,
                                               cudf::mask_state::UNALLOCATED,
                                               cudf::get_default_stream());
  auto const o_shippriority =
    cudf::fill(empty->view(), 0, num_rows, cudf::numeric_scalar<int64_t>(0));

  // Generate the `o_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const o_comment = gen_rand_str_col(19, 78, num_rows);

  // Generate the `lineitem` table. For each row in the `orders` table,
  // we have a random number (between 1 and 7) of rows in the `lineitem` table

  // For each `o_orderkey`, generate a random number (between 1 and 7),
  // which will be the number of rows in the `lineitem` table that will
  // have the same `l_orderkey`
  auto const o_orderkey_repeat_freqs = gen_rand_num_col<int64_t>(1, 7, num_rows);

  // Sum up the `o_orderkey_repeat_freqs` to get the number of rows in the
  // `lineitem` table. This is required to generate the independent columns
  // in the `lineitem` table
  auto const l_num_rows = l_calc_cardinality(o_orderkey_repeat_freqs->view());

  // We create a column, `l_pkey` which will contain the repeated primary keys,
  // `_o_pkey` of the `orders` table as per the frequencies in `o_orderkey_repeat_freqs`
  auto const l_pkey =
    cudf::repeat(cudf::table_view({o_pkey->view()}), o_orderkey_repeat_freqs->view());

  // To generate the base `lineitem` table, we would need to perform a left join
  // between table(o_pkey, o_orderkey, o_orderdate) and table(l_pkey).
  // The column at index 2 in the `l_base` table will comprise the `l_orderkey` column.
  auto const left_table = cudf::table_view({l_pkey->view()});
  auto const right_table =
    cudf::table_view({o_pkey->view(), o_orderkey.view(), o_orderdate_ts->view()});
  auto const l_base_unsorted =
    perform_left_join(left_table, right_table, {0}, {0}, cudf::null_equality::EQUAL);
  auto const l_base = cudf::sort_by_key(l_base_unsorted->view(),
                                        cudf::table_view({l_base_unsorted->get_column(2).view()}));

  // Generate the `l_orderkey` column
  auto const l_orderkey = l_base->get_column(2);

  // Generate the `l_partkey` column
  auto const l_partkey = gen_rand_num_col<int64_t>(1, 200'000 * scale_factor, l_num_rows);

  // Generate the `l_suppkey` column
  auto const l_suppkey = l_calc_suppkey(l_partkey->view(), scale_factor, l_num_rows);

  // Generate the `l_linenumber` column
  auto l_linenumber = gen_rep_seq_col(7, l_num_rows);

  // Generate the `l_quantity` column
  auto const l_quantity = gen_rand_num_col<int64_t>(1, 50, l_num_rows);

  // Generate the `l_discount` column
  auto const l_discount = gen_rand_num_col<double>(0.0, 0.10, l_num_rows);

  // Generate the `l_tax` column
  auto const l_tax = gen_rand_num_col<double>(0.0, 0.08, l_num_rows);

  // NOTE: For now, adding months. Need to add a new `add_calendrical_days` function
  // to add days to the `o_orderdate` column. For implementing this column, we use
  // the column at index 3 in the `l_base` table.
  auto const ol_orderdate_ts = l_base->get_column(3);

  // Generate the `l_shipdate` column
  auto const l_shipdate_rand_add_days = gen_rand_num_col<int32_t>(1, 6, l_num_rows);
  auto const l_shipdate_ts            = cudf::datetime::add_calendrical_months(
    ol_orderdate_ts.view(), l_shipdate_rand_add_days->view());

  auto const d1 = gen_rep_seq_col(1, l_num_rows);
  auto const d1_duration_days =
    cudf::cast(d1->view(), cudf::data_type{cudf::type_id::DURATION_DAYS});
  auto const l_shipdate_ts_plus_x =
    cudf::binary_operation(l_shipdate_ts->view(),
                           d1_duration_days->view(),
                           cudf::binary_operator::ADD,
                           cudf::data_type{cudf::type_id::TIMESTAMP_DAYS});

  // Generate the `l_commitdate` column
  auto const l_commitdate_rand_add_days = gen_rand_num_col<int32_t>(1, 6, l_num_rows);
  auto const l_commitdate_ts            = cudf::datetime::add_calendrical_months(
    ol_orderdate_ts.view(), l_commitdate_rand_add_days->view());

  // Generate the `l_receiptdate` column
  auto const l_receiptdate_rand_add_days = gen_rand_num_col<int32_t>(1, 6, l_num_rows);
  auto const l_receiptdate_ts            = cudf::datetime::add_calendrical_months(
    l_shipdate_ts->view(), l_receiptdate_rand_add_days->view());

  auto current_date =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1995, 6, 17), true);
  auto current_date_literal = cudf::ast::literal(current_date);

  // Generate the `l_returnflag` column
  auto l_receiptdate_col_ref = cudf::ast::column_reference(0);
  auto l_returnflag_pred     = cudf::ast::operation(
    cudf::ast::ast_operator::LESS_EQUAL, l_receiptdate_col_ref, current_date_literal);
  auto l_returnflag_binary_mask =
    cudf::compute_column(cudf::table_view({l_receiptdate_ts->view()}), l_returnflag_pred);
  auto l_returnflag_binary_mask_int =
    cudf::cast(l_returnflag_binary_mask->view(), cudf::data_type{cudf::type_id::INT64});

  auto multiplier                = gen_rep_seq_col(2, l_num_rows);  // 1, 2, 1, 2,...
  auto l_returnflag_ternary_mask = cudf::binary_operation(l_returnflag_binary_mask_int->view(),
                                                          multiplier->view(),
                                                          cudf::binary_operator::MUL,
                                                          cudf::data_type{cudf::type_id::INT64});

  auto l_returnflag_ternary_mask_str =
    cudf::strings::from_integers(l_returnflag_ternary_mask->view());

  auto const l_returnflag_replace_target =
    cudf::test::strings_column_wrapper({"0", "1", "2"}).release();
  auto const l_returnflag_replace_with =
    cudf::test::strings_column_wrapper({"N", "A", "R"}).release();

  auto const l_returnflag = cudf::strings::replace(l_returnflag_ternary_mask_str->view(),
                                                   l_returnflag_replace_target->view(),
                                                   l_returnflag_replace_with->view());

  // Generate the `l_linestatus` column
  auto const l_shipdate_ts_col_ref = cudf::ast::column_reference(0);
  auto const l_linestatus_pred     = cudf::ast::operation(
    cudf::ast::ast_operator::GREATER, l_shipdate_ts_col_ref, current_date_literal);
  auto const l_linestatus_mask =
    cudf::compute_column(cudf::table_view({l_shipdate_ts->view()}), l_linestatus_pred);

  auto const l_linestatus_mask_int =
    cudf::cast(l_linestatus_mask->view(), cudf::data_type{cudf::type_id::INT64});
  auto const l_linestatus_mask_str = cudf::strings::from_integers(l_linestatus_mask_int->view());

  auto const l_linestatus_replace_target = cudf::test::strings_column_wrapper({"0", "1"}).release();
  auto const l_linestatus_replace_with   = cudf::test::strings_column_wrapper({"F", "O"}).release();

  auto const l_linestatus = cudf::strings::replace(l_linestatus_mask_str->view(),
                                                   l_linestatus_replace_target->view(),
                                                   l_linestatus_replace_with->view());

  // Generate the `l_shipinstruct` column
  auto const l_shipinstruct = gen_rand_str_col_from_set(vocab_instructions, l_num_rows);

  // Generate the `l_shipmode` column
  auto const l_shipmode = gen_rand_str_col_from_set(vocab_modes, l_num_rows);

  // Generate the `l_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const l_comment = gen_rand_str_col(10, 43, l_num_rows);

  // Generate the `o_totalprice` column
  auto l_charge   = l_calc_charge(l_tax->view(), l_tax->view(), l_discount->view());
  auto const keys = cudf::table_view({l_orderkey.view()});
  cudf::groupby::groupby gb(keys);
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.push_back(cudf::groupby::aggregation_request());
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  requests[0].values = l_charge->view();
  auto agg_result    = gb.aggregate(requests);
  auto o_totalprice  = std::move(agg_result.second[0].results[0]);

  // Generate the `o_orderstatus` column
  auto const keys2 = cudf::table_view({l_orderkey.view()});
  cudf::groupby::groupby gb2(keys2);
  std::vector<cudf::groupby::aggregation_request> requests2;
  requests2.push_back(cudf::groupby::aggregation_request());

  requests2[0].aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
  requests2[0].values = l_orderkey.view();

  requests2.push_back(cudf::groupby::aggregation_request());
  requests2[1].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  requests2[1].values = l_linestatus_mask_int->view();

  auto agg_result2 = gb2.aggregate(requests2);
  auto const count64 =
    cudf::cast(agg_result2.second[0].results[0]->view(), cudf::data_type{cudf::type_id::INT64});
  auto const ttt = cudf::table_view({agg_result2.first->get_column(0).view(),
                                     count64->view(),
                                     agg_result2.second[1].results[0]->view()});

  write_parquet(ttt, "ttt.parquet", {"l_orderkey", "count", "sum"});

  // if sum == count, then o_orderstatus = 'O'
  // if sum == 0, then o_orderstatus = 'F'
  // else o_orderstatus = 'P'

  auto const count_ref = cudf::ast::column_reference(1);
  auto const sum_ref   = cudf::ast::column_reference(2);
  auto expr            = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, sum_ref, count_ref);
  auto const mask      = cudf::compute_column(ttt, expr);

  auto const col_aa =
    cudf::copy_if_else(cudf::string_scalar("O"), cudf::string_scalar("F"), mask->view());

  auto const ttta = cudf::table_view({agg_result2.first->get_column(0).view(),
                                      count64->view(),
                                      agg_result2.second[1].results[0]->view(),
                                      col_aa->view()});
  write_parquet(ttta, "ttta.parquet", {"l_orderkey", "count", "sum", "o_orderstatus"});

  auto zero_scalar  = cudf::numeric_scalar<int64_t>(0);
  auto zero_literal = cudf::ast::literal(zero_scalar);
  auto expr2_a      = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, sum_ref, count_ref);
  auto expr2_b = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, sum_ref, zero_literal);
  auto expr2   = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr2_a, expr2_b);

  auto const mask2  = cudf::compute_column(ttt, expr2);
  auto const col_bb = cudf::copy_if_else(cudf::string_scalar("P"), col_aa->view(), mask2->view());
  auto const tttaa  = cudf::table_view({agg_result2.first->get_column(0).view(),
                                        count64->view(),
                                        agg_result2.second[1].results[0]->view(),
                                        col_aa->view(),
                                        col_bb->view()});
  write_parquet(
    tttaa, "tttaa.parquet", {"l_orderkey", "count", "sum", "o_orderstatus", "o_orderstatus2"});

  // Write the `orders` table to a parquet file
  auto orders = cudf::table_view({o_orderkey.view(),
                                  o_custkey->view(),
                                  o_totalprice->view(),
                                  o_orderdate_ts->view(),
                                  o_orderpriority->view(),
                                  o_clerk->view(),
                                  o_shippriority->view(),
                                  o_comment->view()});

  write_parquet(orders, "orders.parquet", schema_orders);

  // Write the `lineitem` table to a parquet file
  auto lineitem = cudf::table_view({l_orderkey.view(),
                                    l_partkey->view(),
                                    l_suppkey->view(),
                                    l_linenumber->view(),
                                    l_quantity->view(),
                                    l_discount->view(),
                                    l_tax->view(),
                                    l_shipdate_ts->view(),
                                    l_shipdate_ts_plus_x->view(),
                                    l_commitdate_ts->view(),
                                    l_receiptdate_ts->view(),
                                    l_returnflag->view(),
                                    l_linestatus->view(),
                                    l_shipinstruct->view(),
                                    l_shipmode->view(),
                                    l_comment->view()});

  write_parquet(lineitem, "lineitem.parquet", schema_lineitem);
}

std::unique_ptr<cudf::column> ps_calc_suppkey(cudf::column_view const& ps_partkey,
                                              int64_t const& scale_factor,
                                              int64_t const& num_rows)
{
  // Generating the `s` col
  auto s_empty = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                           num_rows,
                                           cudf::mask_state::UNALLOCATED,
                                           cudf::get_default_stream());

  auto s =
    cudf::fill(s_empty->view(), 0, num_rows, cudf::numeric_scalar<int64_t>(10000 * scale_factor));

  // Generating the `i` col
  auto i = gen_rep_seq_col(4, num_rows);

  // Create a table view out of `p_partkey`, `s`, and `i`
  auto table = cudf::table_view({ps_partkey, s->view(), i->view()});

  // Create the AST expression
  auto scalar_1  = cudf::numeric_scalar<int64_t>(1);
  auto scalar_4  = cudf::numeric_scalar<int64_t>(4);
  auto literal_1 = cudf::ast::literal(scalar_1);
  auto literal_4 = cudf::ast::literal(scalar_4);

  auto ps_partkey_col_ref = cudf::ast::column_reference(0);
  auto s_col_ref          = cudf::ast::column_reference(1);
  auto i_col_ref          = cudf::ast::column_reference(2);

  // (int)(ps_partkey - 1)/s
  auto expr_a = cudf::ast::operation(cudf::ast::ast_operator::SUB, ps_partkey_col_ref, literal_1);
  auto expr_b = cudf::ast::operation(cudf::ast::ast_operator::DIV, expr_a, s_col_ref);
  auto expr_b_casted = cudf::ast::operation(cudf::ast::ast_operator::CAST_TO_INT64, expr_b);

  // s/4
  auto expr_c = cudf::ast::operation(cudf::ast::ast_operator::DIV, s_col_ref, literal_4);

  // (s/4 + (int)(ps_partkey - 1)/s)
  auto expr_d = cudf::ast::operation(cudf::ast::ast_operator::ADD, expr_c, expr_b_casted);

  // (i * (s/4 + (int)(ps_partkey - 1)/s))
  auto expr_e = cudf::ast::operation(cudf::ast::ast_operator::MUL, i_col_ref, expr_d);

  // (ps_partkey + (i * (s/4 + (int)(ps_partkey - 1)/s)))
  auto expr_f = cudf::ast::operation(cudf::ast::ast_operator::ADD, ps_partkey_col_ref, expr_e);

  // (ps_partkey + (i * (s/4 + (int)(ps_partkey - 1)/s))) % s
  auto expr_g = cudf::ast::operation(cudf::ast::ast_operator::MOD, expr_f, s_col_ref);

  // (ps_partkey + (i * (s/4 + (int)(ps_partkey - 1)/s))) % s + 1
  auto final_expr = cudf::ast::operation(cudf::ast::ast_operator::ADD, expr_g, literal_1);

  // Execute the AST expression
  auto ps_suppkey = cudf::compute_column(table, final_expr);
  return ps_suppkey;
}

/**
 * @brief Generate the `partsupp` table
 *
 * @param scale_factor The scale factor to use
 * @param stream The CUDA stream to use
 * @param mr The memory resource to use
 */
void generate_partsupp(int64_t const& scale_factor,
                       rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                       rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  cudf::size_type const num_rows_part = 200'000 * scale_factor;
  cudf::size_type const num_rows      = 800'000 * scale_factor;

  // Generate the `ps_partkey` column
  auto const p_partkey      = gen_primary_key_col(1, num_rows_part);
  auto const rep_freq_empty = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                                        num_rows_part,
                                                        cudf::mask_state::UNALLOCATED,
                                                        cudf::get_default_stream());
  auto const rep_freq =
    cudf::fill(rep_freq_empty->view(), 0, num_rows_part, cudf::numeric_scalar<int64_t>(4));
  auto const rep_table  = cudf::repeat(cudf::table_view({p_partkey->view()}), rep_freq->view());
  auto const ps_partkey = rep_table->get_column(0);

  // Generate the `ps_suppkey` column
  auto const ps_suppkey = ps_calc_suppkey(ps_partkey.view(), scale_factor, num_rows);

  // Generate the `p_availqty` column
  auto const ps_availqty = gen_rand_num_col<int64_t>(1, 9999, num_rows);

  // Generate the `p_supplycost` column
  auto const ps_supplycost = gen_rand_num_col<double>(1.0, 1000.0, num_rows);

  // Generate the `p_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const ps_comment = gen_rand_str_col(49, 198, num_rows);

  auto partsupp = cudf::table_view({ps_partkey.view(),
                                    ps_suppkey->view(),
                                    ps_availqty->view(),
                                    ps_supplycost->view(),
                                    ps_comment->view()});
  write_parquet(partsupp, "partsupp.parquet", schema_partsupp);
}

std::unique_ptr<cudf::column> calc_p_retailprice(cudf::column_view const& p_partkey)
{
  // (
  //            90000
  //            +
  //            (
  //                  (P_PARTKEY/10)
  //                      modulo
  //                       20001
  //            )
  //            +
  //            100
  //            *
  //            (P_PARTKEY modulo 1000)
  // )
  // /100
  auto val_a = cudf::binary_operation(p_partkey,
                                      cudf::numeric_scalar<int64_t>(10),
                                      cudf::binary_operator::DIV,
                                      cudf::data_type{cudf::type_id::FLOAT64});

  auto val_b = cudf::binary_operation(val_a->view(),
                                      cudf::numeric_scalar<int64_t>(20001),
                                      cudf::binary_operator::MOD,
                                      cudf::data_type{cudf::type_id::INT64});

  auto val_c = cudf::binary_operation(p_partkey,
                                      cudf::numeric_scalar<int64_t>(1000),
                                      cudf::binary_operator::MOD,
                                      cudf::data_type{cudf::type_id::INT64});

  auto val_d = cudf::binary_operation(val_c->view(),
                                      cudf::numeric_scalar<int64_t>(100),
                                      cudf::binary_operator::MUL,
                                      cudf::data_type{cudf::type_id::INT64});
  // 90000 + val_b + val_d
  auto val_e = cudf::binary_operation(val_b->view(),
                                      cudf::numeric_scalar<int64_t>(90000),
                                      cudf::binary_operator::ADD,
                                      cudf::data_type{cudf::type_id::INT64});

  auto val_f = cudf::binary_operation(val_e->view(),
                                      val_d->view(),
                                      cudf::binary_operator::ADD,
                                      cudf::data_type{cudf::type_id::INT64});

  auto p_retailprice = cudf::binary_operation(val_f->view(),
                                              cudf::numeric_scalar<int64_t>(100),
                                              cudf::binary_operator::DIV,
                                              cudf::data_type{cudf::type_id::FLOAT64});

  return p_retailprice;
}

/**
 * @brief Generate the `part` table
 *
 * @param scale_factor The scale factor to use
 * @param stream The CUDA stream to use
 * @param mr The memory resource to use
 */
std::unique_ptr<cudf::table> generate_part(
  int64_t const& scale_factor,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  cudf::size_type const num_rows = 200'000 * scale_factor;

  // Generate the `p_partkey` column
  auto const p_partkey = gen_primary_key_col(1, num_rows);

  // Generate the `p_name` column
  auto const p_name_a     = gen_rand_str_col_from_set(vocab_p_name, num_rows);
  auto const p_name_b     = gen_rand_str_col_from_set(vocab_p_name, num_rows);
  auto const p_name_c     = gen_rand_str_col_from_set(vocab_p_name, num_rows);
  auto const p_name_d     = gen_rand_str_col_from_set(vocab_p_name, num_rows);
  auto const p_name_e     = gen_rand_str_col_from_set(vocab_p_name, num_rows);
  auto const p_name_parts = cudf::table_view(
    {p_name_a->view(), p_name_b->view(), p_name_c->view(), p_name_d->view(), p_name_e->view()});
  auto const p_name = cudf::strings::concatenate(p_name_parts, cudf::string_scalar(" "));

  // Generate the `p_mfgr` column
  auto const mfgr_repeat         = gen_rep_str_col("Manufacturer#", num_rows);
  auto const random_values_m     = gen_rand_num_col<int64_t>(1, 5, num_rows);
  auto const random_values_m_str = cudf::strings::from_integers(random_values_m->view());
  auto const p_mfgr              = cudf::strings::concatenate(
    cudf::table_view({mfgr_repeat->view(), random_values_m_str->view()}));

  // Generate the `p_brand` column
  auto const brand_repeat        = gen_rep_str_col("Brand#", num_rows);
  auto const random_values_n     = gen_rand_num_col<int64_t>(1, 5, num_rows);
  auto const random_values_n_str = cudf::strings::from_integers(random_values_n->view());
  auto const p_brand             = cudf::strings::concatenate(cudf::table_view(
    {brand_repeat->view(), random_values_m_str->view(), random_values_n_str->view()}));

  // Generate the `p_type` column
  auto const p_type = gen_rand_str_col_from_set(gen_vocab_types(), num_rows);

  // Generate the `p_size` column
  auto const p_size = gen_rand_num_col<int64_t>(1, 50, num_rows);

  // Generate the `p_container` column
  auto const p_container = gen_rand_str_col_from_set(gen_vocab_containers(), num_rows);

  // Generate the `p_retailprice` column
  auto const p_retailprice = calc_p_retailprice(p_partkey->view());

  // Generate the `p_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const p_comment = gen_rand_str_col(5, 22, num_rows);

  // Create the `part` table
  auto part_view = cudf::table_view({p_partkey->view(),
                                     p_name->view(),
                                     p_mfgr->view(),
                                     p_brand->view(),
                                     p_type->view(),
                                     p_size->view(),
                                     p_container->view(),
                                     p_retailprice->view(),
                                     p_comment->view()});

  return std::make_unique<cudf::table>(part_view);
}

/**
 * @brief Generate the `nation` table
 *
 * @param scale_factor The scale factor to use
 * @param stream The CUDA stream to use
 * @param mr The memory resource to use
 */
void generate_nation(int64_t const& scale_factor,
                     rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                     rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  cudf::size_type const num_rows = 25;

  // Generate the `n_nationkey` column
  auto const n_nationkey = gen_primary_key_col(0, num_rows, stream, mr);

  // Generate the `n_name` column
  auto const n_name = cudf::test::strings_column_wrapper(nations.begin(), nations.end()).release();

  // Generate the `n_regionkey` column
  thrust::host_vector<int64_t> const region_keys     = {0, 1, 1, 1, 4, 0, 3, 3, 2, 2, 4, 4, 2,
                                                        4, 0, 0, 0, 1, 2, 3, 4, 2, 3, 3, 1};
  thrust::device_vector<int64_t> const d_region_keys = region_keys;

  auto n_regionkey = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, num_rows, cudf::mask_state::UNALLOCATED, stream);
  thrust::copy(rmm::exec_policy(stream),
               d_region_keys.begin(),
               d_region_keys.end(),
               n_regionkey->mutable_view().begin<int64_t>());

  // Generate the `n_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const n_comment = gen_rand_str_col(31, 114, num_rows, stream, mr);

  // Create the `nation` table
  auto nation =
    cudf::table_view({n_nationkey->view(), n_name->view(), n_regionkey->view(), n_comment->view()});
  write_parquet(nation, "nation.parquet", schema_nation);
}

/**
 * @brief Generate the `region` table
 *
 * @param scale_factor The scale factor to use
 * @param stream The CUDA stream to use
 * @param mr The memory resource to use
 */
void generate_region(int64_t const& scale_factor,
                     rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                     rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  cudf::size_type const num_rows = 5;

  // Generate the `r_regionkey` column
  auto const r_regionkey = gen_primary_key_col(0, num_rows, stream, mr);

  // Generate the `r_name` column
  auto const r_name =
    cudf::test::strings_column_wrapper({"AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"})
      .release();

  // Generate the `r_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const r_comment = gen_rand_str_col(31, 115, num_rows, stream, mr);

  // Create the `region` table
  auto region = cudf::table_view({r_regionkey->view(), r_name->view(), r_comment->view()});
  write_parquet(region, "region.parquet", schema_region);
}

/**
 * @brief Generate the `customer` table
 *
 * @param scale_factor The scale factor to use
 * @param stream The CUDA stream to use
 * @param mr The memory resource to use
 */
void generate_customer(int64_t const& scale_factor,
                       rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                       rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  cudf::size_type const num_rows = 150'000 * scale_factor;

  // Generate the `c_custkey` column
  auto const c_custkey = gen_primary_key_col(1, num_rows, stream, mr);

  // Generate the `c_name` column
  auto const customer_repeat = gen_rep_str_col("Customer#", num_rows, stream, mr);
  auto const c_custkey_str   = cudf::strings::from_integers(c_custkey->view(), stream, mr);
  auto const c_custkey_str_padded =
    cudf::strings::pad(c_custkey_str->view(), 9, cudf::strings::side_type::LEFT, "0", stream, mr);
  auto const c_name = cudf::strings::concatenate(
    cudf::table_view({customer_repeat->view(), c_custkey_str_padded->view()}), stream, mr);

  // Generate the `c_address` column
  // NOTE: This column is not compliant with clause 4.2.2.7 of the TPC-H specification
  auto const c_address = gen_rand_str_col(10, 40, num_rows, stream, mr);

  // Generate the `c_nationkey` column
  auto const c_nationkey = gen_rand_num_col<int64_t>(0, 24, num_rows, stream, mr);

  // Generate the `c_phone` column
  auto const c_phone = gen_phone_col(num_rows, stream, mr);

  // Generate the `c_acctbal` column
  auto const c_acctbal = gen_rand_num_col<double>(-999.99, 9999.99, num_rows, stream, mr);

  // Generate the `c_mktsegment` column
  auto const c_mktsegment = gen_rand_str_col_from_set(vocab_segments, num_rows, stream, mr);

  // Generate the `c_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const c_comment = gen_rand_str_col(29, 116, num_rows, stream, mr);

  // Create the `customer` table
  auto customer = cudf::table_view({c_custkey->view(),
                                    c_name->view(),
                                    c_address->view(),
                                    c_nationkey->view(),
                                    c_phone->view(),
                                    c_acctbal->view(),
                                    c_mktsegment->view(),
                                    c_comment->view()});
  write_parquet(customer, "customer.parquet", schema_customer);
}

/**
 * @brief Generate the `supplier` table
 *
 * @param scale_factor The scale factor to use
 * @param stream The CUDA stream to use
 * @param mr The memory resource to use
 */
void generate_supplier(int64_t const& scale_factor,
                       rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                       rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  cudf::size_type const num_rows = 10'000 * scale_factor;

  // Generate the `s_suppkey` column
  auto const s_suppkey = gen_primary_key_col(1, num_rows, stream, mr);

  // Generate the `s_name` column
  auto const supplier_repeat = gen_rep_str_col("Supplier#", num_rows, stream, mr);
  auto const s_suppkey_str   = cudf::strings::from_integers(s_suppkey->view(), stream, mr);
  auto const s_suppkey_str_padded =
    cudf::strings::pad(s_suppkey_str->view(), 9, cudf::strings::side_type::LEFT, "0", stream, mr);
  auto const s_name = cudf::strings::concatenate(
    cudf::table_view({supplier_repeat->view(), s_suppkey_str_padded->view()}), stream, mr);

  // Generate the `s_address` column
  // NOTE: This column is not compliant with clause 4.2.2.7 of the TPC-H specification
  auto const s_address = gen_rand_str_col(10, 40, num_rows, stream, mr);

  // Generate the `s_nationkey` column
  auto const s_nationkey = gen_rand_num_col<int64_t>(0, 24, num_rows, stream, mr);

  // Generate the `s_phone` column
  auto const s_phone = gen_phone_col(num_rows, stream, mr);

  // Generate the `s_acctbal` column
  auto const s_acctbal = gen_rand_num_col<double>(-999.99, 9999.99, num_rows, stream, mr);

  // Generate the `s_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const s_comment = gen_rand_str_col(25, 100, num_rows, stream, mr);

  // Create the `supplier` table
  auto supplier = cudf::table_view({s_suppkey->view(),
                                    s_name->view(),
                                    s_address->view(),
                                    s_nationkey->view(),
                                    s_phone->view(),
                                    s_acctbal->view(),
                                    s_comment->view()});
  write_parquet(supplier, "supplier.parquet", schema_supplier);
}

int main(int argc, char** argv)
{
  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::set_current_device_resource(&cuda_mr);

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <scale_factor>" << std::endl;
    return 1;
  }

  int32_t scale_factor = std::atoi(argv[1]);
  std::cout << "Requested scale factor: " << scale_factor << std::endl;

  generate_lineitem_and_orders(scale_factor);
  generate_partsupp(scale_factor);
  generate_supplier(scale_factor);
  generate_customer(scale_factor);
  generate_nation(scale_factor);
  generate_region(scale_factor);

  return 0;
}
