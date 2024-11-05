#!/usr/bin/env python3

import argparse
import json
import os
import re

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(here))


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate container galleries",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data",
        help="data directory",
        default=os.path.join(here, "data"),
    )
    parser.add_argument(
        "--out",
        help="directory to save parsed results",
        default=os.path.join(here, "web", "machines"),
    )
    return parser


def recursive_find(base, pattern="*.*"):
    """
    Recursively find and yield files matching a glob pattern.
    """
    for root, _, filenames in os.walk(base):
        for filename in filenames:
            if not re.search(pattern, filename):
                continue
            yield os.path.join(root, filename)


def find_inputs(input_dir):
    """
    Find inputs (times results files)
    """
    files = []
    for filename in recursive_find(input_dir, pattern="machine.png"):
        # We only have data for small
        files.append(filename)
    return files


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    # Output images and data
    outdir = os.path.abspath(args.out)
    indir = os.path.abspath(args.data)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Find input files (skip anything with test)
    files = find_inputs(indir)
    if not files:
        raise ValueError(f"There are no input files in {indir}")

    # Saves raw data to file
    parse_data(indir, outdir, files)


def read_file(filename):
    with open(filename, "r") as fd:
        content = fd.read()
    return content


def write_json(obj, filename):
    with open(filename, "w") as fd:
        fd.write(json.dumps(obj, indent=4))


def write_file(text, filename):
    with open(filename, "w") as fd:
        fd.write(text)


def parse_data(indir, outdir, files):
    # Keep a lookup of relative file paths from the root.
    lookup = {}

    # It's important to just parse raw data once, and then use intermediate
    for filename in files:
        parts = filename.replace(indir + os.sep, "").split(os.sep)
        prefix = "-".join(parts[0:3])
        size = parts[3]
        # This was just testing
        if prefix == "google-compute-engine-cpu" and size == 2:
            continue
        save_dir = os.path.join(outdir, prefix)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if prefix not in lookup:
            lookup[prefix] = []
        relpath = "../../../" + os.path.relpath(filename, here)
        lookup[prefix].append({"path": relpath, "size": size})

    # Write json data
    for prefix, machines in lookup.items():
        i = 0
        data = {"gallery": []}
        tags = prefix.split("-")
        for machine in machines:
            data["gallery"].append(
                {
                    "id": i,
                    "desc": prefix,
                    "tags": tags + [machine["size"]],
                    "src": machine["path"],
                }
            )
            i += 1
        save_file = os.path.join(outdir, prefix, "data.json")
        write_json(data, save_file)


if __name__ == "__main__":
    main()
