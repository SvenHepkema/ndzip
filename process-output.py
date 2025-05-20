#!/usr/bin/env python3

import os
import sys

import argparse
import logging
import itertools

import inspect
import types
from pathlib import Path
from typing import Iterable

import polars as pl


def directory_exists(path: str) -> bool:
    return Path(path).is_dir()


def get_processing_functions() -> list[tuple[str, types.FunctionType]]:
    current_module = inspect.getmodule(inspect.currentframe())
    functions = inspect.getmembers(current_module, inspect.isfunction)

    processing_function_prefix = "process_"
    processing_functions = filter(
        lambda x: x[0].startswith(processing_function_prefix), functions
    )
    stripped_prefixes_from_name = map(
        lambda x: (x[0].replace(processing_function_prefix, ""), x[1]),
        processing_functions,
    )
    return list(stripped_prefixes_from_name)


def get_all_files_in_dir(dir: str) -> list[str]:
    file_paths = []

    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if os.path.isfile(file_path):
            file_paths.append(file_path)

    return file_paths


def get_all_files_with_prefix_in_dir(dir: str, prefix: str) -> list[str]:
    return list(
        filter(lambda x: x.split("/")[-1].startswith(prefix), get_all_files_in_dir(dir))
    )


def parse_duration(line: str) -> int:
    """
    Parses the time, returns as nanoseconds
    """
    multiplier = None

    duration = line.split()[1]
    unit: str = duration[-2]
    if unit == "u":
        multiplier = 1000
    elif unit == "m":
        multiplier = 1000 * 1000
    else:
        if duration[-2].isnumeric() and duration[-1] == "s":
            multiplier = 1000 * 1000 * 1000
        else:
            raise Exception(f"Could not parse {duration}, unknown unit: {unit}.")

    value_float: float = float(duration[:-2])

    return int(value_float * multiplier)


def duplicate_each(collection: Iterable, repeat_n_times: int) -> list:
    return [x for x in collection for _ in range(repeat_n_times)]


def add_sample_run_column(df: pl.DataFrame, n_samples: int) -> pl.DataFrame:
    assert df.height % n_samples == 0, "Number of samples is not divisible by n_samples"
    n_unique_kernels = df.height // n_samples
    return df.with_columns(
        pl.Series("sample_run", list(range(1, n_samples + 1)) * n_unique_kernels),
    )

def convert_section_to_df(section: str) -> pl.DataFrame:
    compression_ratio = None
    decompression_times = []
    decompression_query_times = []
    file = None
    data_type = None
    algo = None
    n_vecs = None
    n_bytes = None

    for i, line in enumerate(section.split("\n")):
        fields = line.split(';')

        # Warmup run
        if i <= 1:
            continue

        if "[DECOMPRESSION]" in line:
            decompression_times.append(float(fields[1]))
            continue

        if "[DECOMPRESSION_QUERY]" in line:
            decompression_query_times.append(float(fields[1]))
            continue

        if "[COMPRESSION_RATIO]" in line:
            compression_ratio = float(fields[1])
            continue

        # Main header
        if len(fields) >= 2:
            file = fields[0]
            data_type = fields[1]
            algo = fields[3]
            n_bytes = int(fields[-2])
            n_vecs = int(n_bytes) // ((4 if data_type == "float" else 8) * 1024)
            continue

        raise Exception(f"Could not parse {line}")

    dfs = []
 
    for kernel, measurements in zip(["decompression", "decompression_query",], [decompression_times, decompression_query_times,]):
        for i, duration in enumerate(measurements):
            dfs.append(pl.DataFrame([{
                "return_code": 0,
                "avg_bits_per_value": None,
                "avg_exceptions_per_vector": None,
                "kernel": kernel,
                "compressor": "ndzip",
                "file": file,
                "n_bytes": n_bytes,
                "duration_ms": duration,
                "compression_ratio": compression_ratio,
                "data_type": "f64" if data_type == "double" else "f32",
                "n_vecs": n_vecs,
                "sample_run": i + 1,
            }]))

    return pl.concat(dfs)


def collect_sections_into_df(
    sections, convertor_lambda
) -> pl.DataFrame:
    return pl.concat(map(convertor_lambda, sections))


def process_compressors() -> tuple[str, pl.DataFrame]:
    sections = []

    with open("results.out") as f:
        section = ""
        for i, line in enumerate(f):
            if i == 0:
                continue

            section += line

            n_runs = 5
            if i % (2 + 2 + 2 * n_runs) == 0:
                sections.append(section.strip())
                section = ""

    return "ndzip.csv", collect_sections_into_df(
        sections, convert_section_to_df 
    )


def main(args):
    assert directory_exists(args.output_dir)


    default_name, df= process_compressors()
    df.write_csv(os.path.join(args.output_dir, default_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")

    parser.add_argument(
        "output_dir",
        default=None,
        type=str,
        help="directory_to_write_results_to",
    )

    parser.add_argument(
        "-ll",
        "--logging-level",
        type=int,
        default=logging.INFO,
        choices=[logging.CRITICAL, logging.ERROR, logging.INFO, logging.DEBUG],
        help=f"logging level to use: {logging.CRITICAL}=CRITICAL, {logging.ERROR}=ERROR, {logging.INFO}=INFO, "
        + f"{logging.DEBUG}=DEBUG, higher number means less output",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.logging_level)  # filename='program.log',
    logging.info(
        f"Started {os.path.basename(sys.argv[0])} with the following args: {args}"
    )

    main(args)
