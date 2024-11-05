#!/usr/bin/env python3
import argparse
import collections
import json
import os
import re

from matplotlib.ticker import FormatStrFormatter
import matplotlib.pylab as plt
import pandas
import seaborn as sns

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(here))

sns.set_theme(style="whitegrid", palette="pastel")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate tables",
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
    for root, dirnames, filenames in os.walk(base):
        for filename in dirnames:
            if not re.search(pattern, filename):
                continue
            yield os.path.join(root, filename)


def recursive_find_file(base, pattern="*.*"):
    """
    Recursively find and yield files matching a glob pattern.
    """
    for root, dirnames, filenames in os.walk(base):
        for filename in filenames:
            if not re.search(pattern, filename):
                continue
            yield os.path.join(root, filename)


def find_inputs(input_dir):
    """
    Find inputs (metadata about nodes results files)
    """
    files = []
    for filename in recursive_find(input_dir, pattern="node-"):
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
    """
    Parse all input data discovered into output directory.
    """
    # Prepare a pandas data frame that flattens all of the data for a cloud
    # This will take forever to append, so do a dict that we can parse into data frames
    dfs = []
    data = {}
    # For each node, write a manifest. We will save the summary manifest to
    # the node directory, and then loop over them to create a table of counts
    # for each environment.
    total = len(files)
    for i, filename in enumerate(files):
        print(f"Parsing {i} of {total}", end="\r")
        parts = filename.replace(indir + os.sep, "").split(os.sep)
        node = parts[-1]
        prefix = "-".join(parts[0:3])
        size = int(parts[3])
        env_type = parts[2]
        env_name = parts[1]

        # This was just testing
        if "google-compute" in prefix and size == 2:
            continue

        if prefix not in data:
            data[prefix] = {}

        # We will prepare a summary for each node
        summary = {}
        raw_path = os.path.join(filename, "raw")

        # Individual parsed data files
        parsed_path = os.path.join(filename, "processed")
        if not os.path.exists(parsed_path):
            os.makedirs(parsed_path)

        data_files = os.listdir(raw_path)
        data_files = [
            x for x in data_files if not re.search("(machine|parsed-data)", x)
        ]

        hostname = None
        for data_file in data_files:
            if data_file == "cat-proc-cpuinfo":
                result = parse_cpuinfo(os.path.join(raw_path, data_file), parsed_path)
                summary[data_file] = result
                summarize_cpuinfo(data, result, data_file, prefix)
            elif data_file == "hostname":
                hostname = read_file(os.path.join(raw_path, data_file))
            elif data_file == "sysbench-threads-run":
                summary[data_file] = parse_sysbench_threads(
                    os.path.join(raw_path, data_file), parsed_path
                )
            elif data_file == "sysbench-cpu-run":
                summary[data_file] = parse_sysbench_general(
                    os.path.join(raw_path, data_file), parsed_path
                )
            elif data_file == "sysbench-mutex-run":
                summary[data_file] = parse_sysbench_general(
                    os.path.join(raw_path, data_file), parsed_path
                )
            elif data_file == "sysbench-fileio-run-file-test-modeseqwr":
                summary[data_file] = parse_sysbench_threads_file(
                    os.path.join(raw_path, data_file), parsed_path
                )
            elif data_file == "lscpu":
                result = parse_lscpu(os.path.join(raw_path, data_file), parsed_path)
                summary[data_file] = result
                summarize_lscpu(data, result, data_file, prefix)

            elif data_file == "dmidecode":
                result, procs = parse_dmidecode(
                    os.path.join(raw_path, data_file), parsed_path
                )
                summary[data_file] = result
                summary[data_file + "-processors"] = procs

        # When we get here, we have all data for single machine in summary
        for metric, items in summary.items():
            if not metric.startswith("sysbench"):
                continue
            for key, value in items.items():
                # Only add numbers here
                if isinstance(value, (int, float)):
                    dfs.append(
                        {
                            "env": env_name,
                            "experiment": prefix,
                            "size": size,
                            "collection": metric,
                            "metric": key,
                            "node": node,
                            "value": value,
                            "env_type": env_type,
                        }
                    )

    # Plot sysbench data
    plot_sysbench(dfs, outdir)

    # Next we will want to save this to data for a table
    flat = []
    for environment, result in data.items():
        save_file = os.path.join(outdir, environment, "summary-data.json")
        write_json(result, save_file)
        # Create flattened version
        for field, values in result.items():
            for attribute, counts in values.items():
                for value, count in counts.items():
                    flat.append(
                        {
                            "environment": environment,
                            "collection": field,
                            "attribute": attribute,
                            "value": value,
                            "count": count,
                        }
                    )
    table_dir = os.path.join(outdir, "table")
    if not os.path.exists(table_dir):
        os.makedirs(table_dir)
    save_file = os.path.join(table_dir, "data.json")
    write_json(flat, save_file)

    root = os.path.dirname(outdir)
    img_outdir = os.path.join(root, "img")

    # Plot select things - first clock speed.
    # Be consistent about colors and prefixes
    colors = sns.color_palette("hls", 12)
    hexcolors = colors.as_hex()
    colors = {}

    dfs = {}
    files = list(recursive_find_file(indir, pattern="cpuinfo[.]json"))
    # google are all the same, aws are all different, azure are mostly different
    differs = {"aws": 0, "google": 0, "azure": 0}
    totals = {"aws": 0, "google": 0, "azure": 0}
    for i, filename in enumerate(files):
        print(f"Parsing {i} of {total}", end="\r")
        # ['google', 'gke', 'cpu', '256', 'node-0', 'processed', 'cpuinfo.json']
        parts = filename.replace(indir + os.sep, "").split(os.sep)
        prefix = "-".join(parts[0:3])
        size = int(parts[3])
        if size not in dfs:
            dfs[size] = {}
        if prefix not in dfs[size]:
            dfs[size][prefix] = []
        if prefix not in colors:
            colors[prefix] = hexcolors.pop(0)
        env_type = parts[2]
        cpuinfo = json.loads(read_file(filename))
        values = [x["cpu MHz"] for x in cpuinfo if "cpu MHz" in x]
        totals[parts[0]] += 1
        if len(set(values)) > 1:
            print(
                f"{prefix} found system with different clock speeds, this should not happen."
            )
            differs[parts[0]] += 1
        dfs[size][prefix] += values

    for size, subset in dfs.items():
        if size < 32:
            continue
        for prefix, values in subset.items():
            if "cpu" not in prefix:
                continue
            print(f"Plotting prefix {prefix}")
            ax = sns.histplot(
                values, color=colors[prefix], bins="sturges", alpha=0.5, label=prefix
            )

        plt.title(f"CPU Clock Speed Across CPU Environments (size {size})")
        ax.set_xlabel("MHz", fontsize=16)
        ax.set_ylabel("Count", fontsize=16)
        plt.tight_layout()
        plt.legend()
        path = os.path.join(img_outdir, f"clock-speeds-cpu-size-{size}.png")
        plt.savefig(path)
        plt.clf()

    # GPU just one plot
    for size, subset in dfs.items():
        for prefix, values in subset.items():
            if "gpu" not in prefix:
                continue
            print(f"Plotting prefix {prefix}")
            ax = sns.histplot(
                values, color=colors[prefix], bins="sturges", alpha=0.5, label=prefix
            )

        plt.title("CPU Clock Speed Across GPU Environments")
        ax.set_xlabel("MHz", fontsize=16)
        ax.set_ylabel("Count", fontsize=16)
        plt.tight_layout()
        plt.legend()
        path = os.path.join(img_outdir, "clock-speeds-gpu.png")
        plt.savefig(path)
        plt.clf()


def plot_sysbench(dfs, outdir):
    """
    This was an attempt to parse all the summary data, which was too much :)
    """
    # Now parse into data frames we can plot
    df = pandas.DataFrame(dfs)

    # Save images to another directory
    root = os.path.dirname(outdir)
    img_outdir = os.path.join(root, "img")
    plot_count = 0

    # Create a combined collection and key in case we have duplicates within collections
    df["uid"] = df["collection"] + "-" + df["metric"]
    for env_type in df.env_type.unique():
        env_df = df[df.env_type == env_type]
        for metric in env_df.uid.unique():
            subset = env_df[env_df.uid == metric]
            colors = sns.color_palette("hls", len(subset.experiment.unique()))
            hexcolors = colors.as_hex()
            types = list(subset.experiment.unique())
            palette = collections.OrderedDict()
            for t in types:
                palette[t] = hexcolors.pop(0)

            # The second case happens sometimes - all values are zero, or the same
            # There are others that produce empty looking plots that need looking into
            if (
                subset.shape[0] == 0
                or subset.value.sum() == 0
                or len(subset.value.unique()) == 1
            ):
                continue
            make_plot(
                subset,
                title=f"Single Node Benchmark {metric} ({env_type})",
                ydimension="value",
                plotname=f"{metric}-{env_type}",
                xdimension="size",
                palette=palette,
                outdir=img_outdir,
                hue="experiment",
                xlabel="Size",
                ylabel=metric,
                do_round=True,
            )
            plot_count += 1


def summarize_lscpu(data, items, collector, prefix):
    """
    Save a subset of fields from lscpu
    """
    if collector not in data[prefix]:
        data[prefix][collector] = {}
    keepers = [
        "Architecture",
        "CPU op-mode(s)",
        "Address sizes",
        "Byte Order",
        "CPU(s)",
        "On-line CPU(s) list",
        "Vendor ID",
        "Model name",
        "CPU family",
        "Model",
        "Thread(s) per core",
        "Core(s) per socket",
        "Socket(s)",
        "Stepping",
        "BogoMIPS",
        "Hypervisor vendor",
        "Virtualization type",
        "NUMA node(s)",
    ]
    for field in keepers:
        value = items[field]
        field = field.replace(" ", "_").lower()
        if field not in data[prefix][collector]:
            data[prefix][collector][field] = {}
        if value not in data[prefix][collector][field]:
            data[prefix][collector][field][value] = 0
        data[prefix][collector][field][value] += 1


def summarize_dmidecode(data, items, collector, prefix):
    """
    Add a node to a global data for dmidecode
    """
    if collector not in data[prefix]:
        data[prefix][collector] = {}
    for field, values in items.items():
        for value, _ in values.items():
            flattened = (
                field.replace(" ", "_").lower() + "_" + value.replace(" ", "_").lower()
            )
            if flattened not in data[prefix][collector]:
                data[prefix][collector][value] = 0
            data[prefix][collector][value] += 1


def summarize_cpuinfo(data, items, collector, prefix):
    """
    Add a node to a global data for cpuinfo
    """
    if collector not in data[prefix]:
        data[prefix][collector] = {}
    # We know these are uniform for the node
    for field, values in items.items():
        for key, _ in values.items():
            # e.g., cpu family, model
            field = field.replace(" ", "_")
            if not isinstance(key, str):
                key = str(key)
            flattened = field.lower()
            # Skip flags for now - too many for a table
            if flattened in ["flags", "bugs"]:
                continue
            if flattened not in data[prefix][collector]:
                data[prefix][collector][flattened] = {}
            if key not in data[prefix][collector][flattened]:
                data[prefix][collector][flattened][key] = 0
            data[prefix][collector][flattened][key] += 1


def parse_clean_line(line):
    """
    Split a line by the : delimiter, strip and clean it.
    """
    key, value = line.split(":", 1)
    key = key.strip()
    value = value.strip()
    return key, value


def untab_line(line):
    """
    Remove tab and extra space, found in dmidecode
    """
    return line.strip("\t").strip()


def parse_dmidecode(filename, parsed_path):
    """
    Parse the output of dmidecode
    """
    info = read_file(filename)
    lines = info.split("\n")
    data = {}

    # There is one entry for everything except for processors
    count = 0

    # Keep a count of processors stuffs
    procs = {}
    while lines:
        line = untab_line(lines.pop(0))
        if line.startswith("BIOS Information"):
            # Same pattern until characteristics
            while "Characteristics" not in line:
                line = untab_line(lines.pop(0))
                key, value = parse_clean_line(line)
                data[f"bios_{key}"] = value
            while True:
                line = untab_line(lines.pop(0))
                if "BIOS Revision" in line:
                    break
                if "is supported" in line:
                    key = key.replace(" is supported", "").strip()
                    data[f"bios_{key}"] = "is supported"
                elif "not supported" in line:
                    key = key.replace(" not supported", "").strip()
                    data[f"bios_{key}"] = "not supported"
                else:
                    if "bios_extra" not in data:
                        data["bios_extra"] = []
                    data["bios_extra"].append(line)

            # Last line is revision (line already popped)
            key, value = parse_clean_line(line)
            data[f"bios_{key}"] = value

        elif line.startswith("System Information"):
            while True:
                line = untab_line(lines.pop(0))
                if not line:
                    break
                key, value = parse_clean_line(line)
                data[f"system_{key}"] = value

        elif line.startswith("Base Board Information"):
            while True:
                line = untab_line(lines.pop(0))
                if not line:
                    break
                if "Features" in line:
                    features = []
                    while "Location" not in line:
                        features.append(line)
                        line = untab_line(lines.pop(0))
                    data["baseboard_features"] = features
                # Line is already popped here
                key, value = parse_clean_line(line)
                data[f"baseboard_{key}"] = value

        elif line.startswith("Chassis Information"):
            while True:
                line = untab_line(lines.pop(0))
                if not line:
                    break
                key, value = parse_clean_line(line)
                data[f"chassis_{key}"] = value

        elif line.startswith("Processor Information"):
            while True:
                line = untab_line(lines.pop(0))
                if not line:
                    break
                # Skip flags, characteristics for now
                if ":" not in line:
                    continue
                key, value = parse_clean_line(line)
                if not value:
                    continue
                data[f"processor_{count}_{key}"] = value
                # Socket designation is always unique to processor
                if key in [
                    "Socket Designation",
                    "L1 Cache Handle",
                    "L2 Cache Handle",
                    "L3 Cache Handle",
                    "Core Enabled",
                ]:
                    continue
                if key not in procs:
                    procs[key] = {}
                if value not in procs[key]:
                    procs[key][value] = 0
                procs[key][value] += 1
            count += 1

    # The counts should all be the same!
    value = None
    for field, values in procs.items():
        for _, count in values.items():
            if value is None:
                value = count
            if value != count:
                print(filename)
                print(field)
                print(values)
                raise ValueError(f"Found processor unlike the others: {procs}")

    save_file = os.path.join(parsed_path, "dmidecode.json")
    write_json(data, save_file)
    save_file = os.path.join(parsed_path, "dmidecode-processors.json")
    write_json(procs, save_file)
    return data, procs


def parse_lscpu(filename, parsed_path):
    """
    Parse the output of lscpu
    """
    info = read_file(filename)
    int_fields = [
        "CPU(s)",
        "CPU family",
        "Model",
        "Thread(s) per core",
        "Core(s) per socket",
        "Socket(s)",
        "Stepping",
        "NUMA node(s)",
    ]

    float_fields = ["BogoMIPS"]
    data = {}
    for line in info.split("\n"):
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "Flags":
            for flag in value.split(" "):
                data[f"flag_{flag}"] = "on"
            continue
        if key in int_fields:
            value = int(value)
        if key in float_fields:
            value = float(value)
        data[key] = value

    # Save unique data to file
    save_file = os.path.join(parsed_path, "lscpu.json")
    write_json(data, save_file)
    return data


def parse_cpuinfo(filename, parsed_path):
    """
    Parse the output (contents) of /proc/cpuinfo
    """
    # These get converted to float or int
    convert_to_int = [
        "processor",
        "cpu family",
        "model",
        "stepping",
        "physical id",
        "siblings",
        "core id",
        "cpu cores",
        "apicid",
        "initial apicid",
        "cpuid level",
        "clflush size",
        "cache alignment",
    ]
    convert_to_float = ["cpu MHz", "bogomips"]
    list_split = ["flags", "bugs"]

    # Assemble processors first
    processor = {}
    info = read_file(filename)
    procs = []
    for line in info.split("\n"):
        if "\t:" not in line:
            continue
        if "processor" in line:
            procs.append(processor)
            processor = {}
        key, value = line.split("\t:", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            continue
        if key in convert_to_int:
            value = int(value)
        if key in convert_to_float:
            value = float(value)
        if key in list_split:
            value = value.split(" ")
        processor[key] = value

    # Save unique data to file
    save_file = os.path.join(parsed_path, "cpuinfo.json")
    write_json(procs, save_file)

    # Create a summary count for each
    counts = {}
    for proc in procs:
        for key, value in proc.items():
            # Skip processor id
            if key in [
                "processor",
                "apicid",
                "initial apicid",
                "coreid",
                "core id",
                "physical id",
                "cpu MHz",
                "bogomips",
            ]:
                continue
            if key not in counts:
                counts[key] = {}
            # flags and bugs are the only two lists
            if isinstance(value, list):
                for v in value:
                    if v not in counts[key]:
                        counts[key][v] = 0
                    counts[key][v] += 1
            else:
                if value not in counts[key]:
                    counts[key][value] = 0
                counts[key][value] += 1

    # This checks if the single node is uniform - it should be
    # We can call a single file uniform if the values have the same count
    count = None
    for attribute, values in counts.items():
        for key, value in values.items():
            if count is None:
                count = value
            if value != count:
                raise ValueError(
                    f"Found single node with different cpu: {attribute} found {values}"
                )

    save_file = os.path.join(parsed_path, "cpuinfo-counts.json")
    write_json(counts, save_file)
    return counts


# Parsing functions assuming a line with some label: value
def parse_float(line):
    return float(line.split(":", 1)[-1].strip())


def parse_int(line):
    return float(line.split(":", 1)[-1].strip())


def parse_line(line):
    return line.split(":", 1)[-1].strip()


def parse_latency(lines):
    return {
        "latency_ms_min": parse_float(lines.pop(0)),
        "latency_ms_avg": parse_float(lines.pop(0)),
        "latency_ms_max": parse_float(lines.pop(0)),
        "latency_ms_95th_percentile": parse_float(lines.pop(0)),
        "latency_ms_sum": parse_float(lines.pop(0)),
    }


def parse_sysbench_threads(filename, parsed_path):
    """
    Parse output of running sysbench for threads
    """
    info = read_file(filename)
    data = {}
    lines = info.split("\n")
    while lines:
        line = lines.pop(0)
        if "Number of threads" in line:
            data["number_of_threads"] = parse_int(line)

        # General statistics:
        #    total time:                          10.0004s
        #    total number of events:              14135

        elif line.startswith("General statistics"):
            data.update(parse_stats(lines))

        # Latency (ms):
        #          min:                                    0.69
        #          avg:                                    0.71
        #          max:                                    0.89
        #          95th percentile:                        0.72
        #          sum:                                 9998.16
        elif line.startswith("Latency"):
            data.update(parse_latency(lines))

        # Threads fairness:
        #    events (avg/stddev):           14135.0000/0.00
        #    execution time (avg/stddev):   9.9982/0.00

        elif line.startswith("Threads fairness"):
            data.update(parse_thread_fairness(lines))

    return data


def parse_sysbench_general(filename, parsed_path):
    """
    Parse the output of running sysbench for cpu or mutex
    """
    info = read_file(filename)
    data = {}
    lines = info.split("\n")
    while lines:
        line = lines.pop(0)
        if "Number of threads" in line:
            data["number_of_threads"] = parse_int(line)

        elif "Prime numbers limit" in line:
            data["prime_numbers_limit"] = parse_int(line)

        elif line.startswith("CPU speed"):
            data["cpu_speed_events_per_second"] = parse_float(lines.pop(0))

        elif line.startswith("General statistics"):
            data.update(parse_stats(lines))

        elif line.startswith("Latency"):
            data.update(parse_latency(lines))

        elif line.startswith("Threads fairness"):
            data.update(parse_thread_fairness(lines))
    return data


def parse_stats(lines):
    return {
        "total_time": parse_line(lines.pop(0)),
        "total_number_events": parse_int(lines.pop(0)),
    }


def parse_thread_fairness(lines):
    return {
        "threads_fairness_events_avg_over_stddev": parse_line(lines.pop(0)),
        "threads_fairness_execution_time_avg_over_stddev": parse_line(lines.pop(0)),
    }


def parse_sysbench_threads_file(filename, parsed_path):
    """
    Parse the output of running sysbench for fileio
    """
    info = read_file(filename)
    data = {}
    lines = info.split("\n")
    while lines:
        line = lines.pop(0)
        if "Number of threads" in line:
            data["number_of_threads"] = parse_int(line)

        if "files," in line:
            data["file_stats"] = line.strip()
        if "total file size" in line:
            data["total_file_size"] = line.split(" ", 1)[0].strip()
        if "Block size" in line:
            data["block_size"] = line.split(" ")[-1].strip()

        # Not included, consistent for the test
        # Periodic FSYNC enabled, calling fsync() each 100 requests.
        # Calling fsync() at the end of test, Enabled.
        # Using synchronous I/O mode
        # Doing sequential write (creation) test
        elif line.startswith("General statistics"):
            data.update(parse_stats(lines))

        elif line.startswith("Latency"):
            data.update(parse_latency(lines))

        elif line.startswith("Threads fairness"):
            data.update(parse_thread_fairness(lines))

        elif line.startswith("File operations"):
            data["reads_per_second"] = parse_float(lines.pop(0))
            data["writes_per_second"] = parse_float(lines.pop(0))
            data["fsyncs_per_second"] = parse_float(lines.pop(0))

        elif line.startswith("Throughput"):
            data["reads_mib_per_second"] = parse_float(lines.pop(0))
            data["written_mib_per_second"] = parse_float(lines.pop(0))
    return data


def make_plot(
    df,
    title,
    ydimension,
    xdimension,
    xlabel,
    ylabel,
    palette=None,
    ext="png",
    plotname="lammps",
    hue=None,
    outdir="img",
    log_scale=False,
    do_round=False,
):
    """
    Helper function to make common plots.
    """
    ext = ext.strip(".")
    plt.figure(figsize=(12, 6))
    sns.set_style("dark")

    flierprops = dict(
        marker=".", markerfacecolor="None", markersize=10, markeredgecolor="black"
    )
    ax = sns.boxplot(
        x=xdimension,
        y=ydimension,
        flierprops=flierprops,
        hue=hue,
        data=df,
        # gap=.1,
        linewidth=0.0,
        palette=palette,
        whis=[5, 95],
        # dodge=False,
    )

    plt.title(title)
    print(log_scale)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=14)
    ax.set_yticklabels(ax.get_yticks(), fontsize=14)
    # plt.xticks(rotation=90)
    if log_scale is True:
        plt.gca().yaxis.set_major_formatter(
            plt.ScalarFormatter(useOffset=True, useMathText=True)
        )

    if do_round is True:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    plt.tight_layout()
    path = os.path.join(outdir, f"{plotname}.{ext}")
    plt.savefig(path)
    plt.clf()
    return path


if __name__ == "__main__":
    main()
