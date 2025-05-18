#!/usr/bin/env python3

import os
import sys

import argparse
import logging

from pathlib import Path

DEFAULT_N_VECS = 25600
VALUES_PER_VEC = 1024

def directory_exists(path: str) -> bool:
    return Path(path).is_dir()


def get_all_files_in_dir(dir: str):
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if os.path.isfile(file_path):
            yield file_path


def get_n_values_file(path: str, data_type: str) -> int:
    file_size = os.path.getsize(path)
    data_type_size = 4 if data_type == "float" else 8
    assert (
        file_size % data_type_size == 0
    ), "file size is not a multiple of data_type size"
    return file_size // data_type_size


def log_data_file(out_file, path: str, data_type: str, n_values: int) -> None:
    line = f"{path};{data_type};{n_values}\n"
    logging.info(line.strip())
    out_file.write(line)


def main(args):
    for file in get_all_files_in_dir(args.input_dir):
        n_values = get_n_values_file(file, args.data_type)
        #n_values = args.n_vecs * VALUES_PER_VEC
        log_data_file(args.out_file, file, args.data_type, n_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="program")
    parser.add_argument(
        "input_dir",
        type=str,
        help="directory with data files",
    )
    parser.add_argument(
        "data_type",
        type=str,
        choices=["float", "double"],
        help="data type",
    )
    parser.add_argument("out_file", type=argparse.FileType("w"), help="Output file")
    parser.add_argument("-nv", "--n-vecs", type=int, default=DEFAULT_N_VECS, help="N vecs input")
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
