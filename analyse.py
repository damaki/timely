import argparse
import csv
import concurrent.futures
import numpy as np
import pandas as pd
import progress.bar
import scipy.stats
import sys
import tempfile
from contextlib import contextmanager
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Dict, Tuple, Optional
from pathlib import Path


class BinarySamplesFileLoader:
    def __init__(self, path: Path, nsamples: int, nclasses: int, dtype: np.dtype):
        self.file_path = path
        self.nsamples = nsamples
        self.nclasses = nclasses
        self.dtype = dtype

    @contextmanager
    def load_samples(self) -> np.ndarray:
        """Load the binary samples file into memory

        Returns
        -------
        A 2D numpy array containing the samples, with one column per input
        data class.
        """

        samples = np.memmap(
            self.file_path,
            mode="r",
            dtype=self.dtype,
            shape=(self.nsamples, self.nclasses),
            order="C",
        )

        yield samples

        # Force an unmap to ensure that the underlying file is properly closed.
        # This ensures all file handles are released when we want to delete the
        # temporary file.
        #
        # Note that any further accesses to the memmapped data will cause the
        # interpreter to crash, so we delete 'samples' to reduce this risk.
        samples._mmap.close()
        del samples


def _convert_samples_csv_to_bin(
    samples_csv_path: Path, samples_bin_path: Path, dtype=np.float64
) -> BinarySamplesFileLoader:
    """Converts the timing samples CSV file to a file in binary format.

    This allows the samples to be memory mapped into multiple worker processes
    and therefore avoid excessive memory usage when dealing with large sample
    sets and many CPU cores.

    Parameters
    ----------
    samples_csv_path
        The path to the CSV file containing the timing samples.
    samples_bin_path
        The path to the file to generate that will contain the samples in
        binary format.
    dtype
        The data type to use for the samples.

    Returns
    -------
    BinarySamplesFileLoader
        Object that can be used to load the samples from the binary file.
    """

    # Determine the number of columns from the first row
    with pd.read_csv(samples_csv_path, chunksize=1, nrows=1, header=None) as csv_reader:
        ncols = len(csv_reader.read(1).columns)

    # Load the CSV file in chunks to limit the peak memory usage
    with pd.read_csv(samples_csv_path, chunksize=512000, dtype=dtype) as csv_reader:
        nrows = 0
        for chunk in csv_reader:
            nrows_in_chunk = len(chunk.index)

            with open(samples_bin_path, "rb+" if nrows > 0 else "wb+") as file:
                samples_bin = np.memmap(
                    file,
                    mode="r+" if nrows > 0 else "w+",
                    dtype=dtype,
                    shape=(nrows + nrows_in_chunk, ncols),
                    order="C",
                )

                # Remove any rows that contain NaNs. This happens when the CSV
                # file contains incomplete/partial rows.
                chunk = chunk.iloc[:, :].values
                chunk = chunk[~np.isnan(chunk).any(axis=1), :]

                samples_bin[nrows:, :] = chunk
                del samples_bin

            nrows += nrows_in_chunk

    return BinarySamplesFileLoader(
        path=samples_bin_path, nsamples=nrows, nclasses=ncols, dtype=dtype
    )


def _class_pair_statistics(samples_bin: BinarySamplesFileLoader, c1: int, c2: int):
    """Run statistical tests to compare two classes of input data

    Parameters
    ----------
    samples_bin
        Load the binary sample data.
    c1
        Index of the first class to analyse
    c2
        Index of the second class to analyse

    Returns
    -------
    dict
        A dictionary containing the results of each statistical test.
    """
    results = {}

    with samples_bin.load_samples() as samples:
        c1_samples = samples[:, c1]
        c2_samples = samples[:, c2]

        results["n"] = len(c1_samples)

        # Wilcoxon signed-rank test
        results["wilcoxon"] = scipy.stats.wilcoxon(c1_samples, c2_samples)

        # t-test on two related samples
        results["ttest"] = scipy.stats.ttest_rel(c1_samples, c2_samples)

        # Sign test on paired samples
        differences = c2_samples - c1_samples
        results["sign_test"] = scipy.stats.binomtest(
            sum(differences < 0.0),
            sum(differences != 0.0),
            p=0.5,
            alternative="two-sided",
        )

        # Statistics on the differences between samples
        quantiles = np.quantile(differences, [0.25, 0.5, 0.75])
        results["differences"] = {
            "mean": np.mean(differences),
            "std": np.std(differences),
            "median": quantiles[1],
            "iqr": quantiles[2] - quantiles[0],
            "mad": scipy.stats.median_abs_deviation(differences),
        }

        return results


def _friedman_test(samples_bin: BinarySamplesFileLoader):
    """Run the Friedman test

    Parameters
    ----------
    samples_bin
        File containing the samples in binary format.

    """
    with samples_bin.load_samples() as samples:
        nclasses = samples.shape[1]
        return scipy.stats.friedmanchisquare(*(samples[:, i] for i in range(nclasses)))


def _draw_samples_scatter_plot(filename: Path, samples_bin: BinarySamplesFileLoader):
    """Generate a scatter plot of all samples from all classes.

    Parameters
    ----------
    filename
        Path to the PNG file to generate.

    samples_bin
        File containing the samples in binary format.
    """
    with samples_bin.load_samples() as samples:
        nclasses = samples.shape[1]

        fig = Figure(figsize=(16, 12))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(samples, ".", fillstyle="none", alpha=0.6)

        ax.set_title("Samples scatter plot")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Time")
        ax.set_yscale("log")

        header = list(range(nclasses))
        fig.legend(header, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.15))

        canvas.print_figure(filename, bbox_inches="tight")


def _draw_class_means_plot(filename: Path, samples_bin: BinarySamplesFileLoader):
    """Generate a plot of the means of each input data class.

    Parameters
    ----------
    filename
        Path to the PNG file to generate.

    samples_bin
        File containing the samples in binary format.
    """
    with samples_bin.load_samples() as samples:
        nclasses = samples.shape[1]

        fig = Figure(figsize=(16, 12))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([np.mean(samples[:, c]) for c in range(nclasses)], ".", alpha=0.6)

        ax.set_title("Class means")
        ax.set_xlabel("Class index")
        ax.set_xticks(list(range(nclasses)))
        ax.set_ylabel("Time")
        ax.set_yscale("log")
        ax.set_ylim(auto=False)

        header = list(range(nclasses))
        fig.legend(header, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.15))

        canvas.print_figure(filename, bbox_inches="tight")


def _draw_classes_box_plot(filename: Path, samples_bin: BinarySamplesFileLoader):
    """Generate a box plot of each input data class.

    Parameters
    ----------
    filename
        Path to the PNG file to generate.

    samples_bin
        File containing the samples in binary format.
    """
    with samples_bin.load_samples() as samples:
        nclasses = samples.shape[1]

        fig = Figure(figsize=(16, 12))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)

        ax.set_title("Samples box plot")
        ax.set_xlabel("Class index")
        ax.set_ylabel("Time")
        ax.set_yscale("log")
        percentiles = np.quantile(
            samples, [0.05, 0.25, 0.5, 0.75, 0.95], overwrite_input=False, axis=0
        )
        percentiles = pd.DataFrame(
            percentiles, columns=list(range(samples.shape[1])), copy=False
        )
        boxes = []
        for name in percentiles:
            vals = [i for i in percentiles.loc[:, name]]
            boxes.append(
                {
                    "label": name,
                    "whislo": vals[0],
                    "q1": vals[1],
                    "med": vals[2],
                    "q3": vals[3],
                    "whishi": vals[4],
                    "fliers": [],
                }
            )

        ax.bxp(boxes, showfliers=False)
        ax.set_xticks(list(range(nclasses + 1)))
        ax.set_xticklabels([""] + list(range(nclasses)))

        header = list(range(nclasses))
        fig.legend(header, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.15))

        canvas.print_figure(filename, bbox_inches="tight")


def _generate_class_pair_csv_report(
    filename: Path,
    class_pair_futures: Dict[Tuple[int, int], concurrent.futures.Future],
):
    """Write each class's statistics to a CSV file

    Parameters
    ----------
    filename
        The path to the CSV file to create
    class_pair_futures
        Maps a pair of class indices to the Future that contains the statistics
        comparing the two classes.
    """
    with open(filename, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "First class",
                "Second class",
                "Wilcoxon signed-rank test p-value",
                "t-test p-value",
                "sign test p-value",
                "Mean of differences",
                "Std. dev. of differences",
                "Median of differences",
                "IQR of differences",
                "Median abs. deviation of differences",
            ]
        )

        for (c1, c2), future in class_pair_futures.items():
            results = future.result()  # blocks until the computation is finished
            writer.writerow(
                [
                    f"class {c1}",
                    f"class {c2}",
                    results["wilcoxon"].pvalue,
                    results["ttest"].pvalue,
                    results["sign_test"].pvalue,
                    results["differences"]["mean"],
                    results["differences"]["std"],
                    results["differences"]["median"],
                    results["differences"]["iqr"],
                    results["differences"]["mad"],
                ]
            )


def _worst_pair(
    class_pair_futures: Dict[Tuple[int, int], concurrent.futures.Future]
) -> Tuple[Tuple[int, int], Dict]:
    """Find the pair of classes that has the lowest p-value from the Wilcoxon
    signed-rank test.

    Parameters
    ----------
    class_pair_futures
        Maps a pair of class indices to the Future that contains the statistics
        comparing the two classes.

    Returns
    -------
    Tuple
        The first item is the pair of class indicies with the biggest difference
        The second item is a dictionary containing the statistics.
    """
    worst_results = None
    worst_pair = None
    for (c1, c2), future in class_pair_futures.items():
        results = future.result()
        if (
            worst_results is None
            or results["wilcoxon"].pvalue < worst_results["wilcoxon"].pvalue
        ):
            worst_results = results
            worst_pair = (c1, c2)

    return worst_pair, worst_results


def _timing_leakage_detected(
    friedman_future: Optional[concurrent.futures.Future],
    class_pair_futures: Dict[Tuple[int, int], concurrent.futures.Future],
    alpha: float,
) -> bool:
    """Check for statistically significant evidence of timing leakage.

    Returns
    -------
    True if evidence of timing leakage was found, or False otherwise.
    """

    if friedman_future is not None and friedman_future.result().pvalue < alpha:
        return True

    for future in class_pair_futures.values():
        results = future.result()

        wilcoxon = results["wilcoxon"]
        sign_test = results["sign_test"]
        ttest = results["ttest"]
        n = results["n"]

        # Apply the Bonferroni correction
        if (
            wilcoxon.pvalue < alpha / n
            or sign_test.pvalue < alpha / n
            or ttest.pvalue < alpha / n
        ):
            return True


def _print_summary(
    class_pair_futures: Dict[Tuple[int, int], concurrent.futures.Future],
    friedman_future: Optional[concurrent.futures.Future],
    alpha: float,
):
    """Print a summary of the results to the standard output

    Parameters
    ----------
    class_pair_futures
        Maps a pair of class indices to the Future that contains the statistics
        comparing the two classes.
    friedman_future
        The Future that contains the result of the Friedman test. This is None
        if the Friedman test was skipped.
    alpha
        The threshold for statistical significance.

    Returns
    -------
    True if evidence of timing leakage was found, or False otherwise.
    """
    worst_class_pair, worst_results = _worst_pair(class_pair_futures)

    if friedman_future is not None:
        print(f"Friedman test p-value: {friedman_future.result().pvalue}")

    print(f"Worst pair: class {worst_class_pair[0]} and class {worst_class_pair[1]}")
    print(f"\tWilcoxon signed-rank test p-value: {worst_results['wilcoxon'].pvalue}")
    print(f"\tt-test p-value: {worst_results['ttest'].pvalue}")
    print(f"\tsign test p-value: {worst_results['sign_test'].pvalue}")

    has_leakage = _timing_leakage_detected(friedman_future, class_pair_futures, alpha)
    print()
    if has_leakage:
        print("Result: Timing leakage detected. Implementation is VULNERABLE.")
    else:
        print("Result: Timing leakage not found. Code *might* be constant time.")

    return has_leakage


def analyse(
    samples_bin: BinarySamplesFileLoader,
    output_dir: Optional[Path],
    alpha: float = 1e-5,
    skip_friedman: bool = False,
    max_workers: Optional[int] = None,
):
    """Run all statistical tests.

    The statistical tests are run in multiple worker processes to take advantage
    of multiple cores to speed up the analysis.

    Output files containing the various results are written to the output_dir
    directory.

    Parameters
    ----------
    samples_bin
        File containing the samples in binary format.
    output_dir
        Directory where the output files should be generated.
    alpha
        The threshold for statistical significance.
    skip_friedman
        When True, don't run the Friedman test.

    Returns
    -------
    True if evidence of timing leakage was found, or False otherwise.
    """
    nsamples = samples_bin.nsamples
    nclasses = samples_bin.nclasses

    progbar = progress.bar.Bar("Processing...", max=0)

    # Some analyses can take a considerable amount of time, so we run them
    # in parallel to take advantage of multiple cores. Separate processes
    # are used to avoid contention with the Python GIL.

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as proc_pool:

        def submit_proc_pool_job(fn, *args, **kwargs):
            """Submit a job to the process pool and advance the progress bar
            when the job is done.
            """
            future = proc_pool.submit(fn, *args, **kwargs)
            progbar.max += 1
            future.add_done_callback(lambda _: progbar.next())
            return future

        # A thread pool is used for most things that need to wait on results
        # (futures) from the process pool since process pool jobs can't
        # block on futures without the risk of deadlocks.

        with concurrent.futures.ThreadPoolExecutor() as thread_pool:
            # scipy's Friedman test uses the chi square approximation which
            # requires at least 3 sets of samples and more than 10 samples in
            # each set, so we skip this test if these conditions aren't met.
            if (not skip_friedman) and nsamples > 10 and nclasses >= 3:
                friedman_future = submit_proc_pool_job(_friedman_test, samples_bin)
            else:
                friedman_future = None

            # Compare all pairs of classes
            class_pair_futures = {}
            for c1 in range(nclasses):
                for c2 in range(c1 + 1, nclasses):
                    class_pair_futures[(c1, c2)] = submit_proc_pool_job(
                        _class_pair_statistics, samples_bin, c1, c2
                    )

            submit_proc_pool_job(
                _draw_samples_scatter_plot, output_dir / "scatter_plot.png", samples_bin
            )
            submit_proc_pool_job(
                _draw_classes_box_plot, output_dir / "box_plot.png", samples_bin
            )
            submit_proc_pool_job(
                _draw_class_means_plot, output_dir / "class_means.png", samples_bin
            )

            thread_pool.submit(
                _generate_class_pair_csv_report,
                output_dir / "class_pair_stats.csv",
                class_pair_futures,
            )

            # Wait for everything to finish
            thread_pool.shutdown(wait=True)
            proc_pool.shutdown(wait=True)

            # The progress bar doesn't put a newline at its end, so we put one here
            print()
            return _print_summary(class_pair_futures, friedman_future, alpha)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-friedman", action="store_true", default=False, help="Skip Friedman test"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        nargs=1,
        type=Path,
        default=None,
        required=True,
        help="Output directory to store results",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        nargs="?",
        type=int,
        default=None,
        help="Specifies the number of jobs to run simultaneously",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        nargs="?",
        type=float,
        default=1e-5,
        help="Threshold for probability of a false positive",
    )
    parser.add_argument(
        "FILE",
        nargs=1,
        type=Path,
        default=None,
        help="CSV file containing the timing measurements to analyse",
    )

    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        print(f"Loading samples from {args.FILE[0]} ...", end=None, flush=True)
        samples_bin_path = tmpdir_path / "samples.bin"
        samples_bin = _convert_samples_csv_to_bin(args.FILE[0], samples_bin_path)
        print("Done")

        print(
            (
                f"Loaded {samples_bin.nclasses} classes "
                f"and {samples_bin.nsamples} samples per class "
                f"({samples_bin.nclasses * samples_bin.nsamples} samples total)"
            )
        )

        output_dir = Path(args.output_dir[0])
        output_dir.mkdir(parents=True, exist_ok=True)

        has_leakage = analyse(
            samples_bin,
            output_dir=output_dir,
            skip_friedman=args.no_friedman,
            alpha=args.alpha,
            max_workers=args.jobs,
        )

        if has_leakage:
            sys.exit(-1)


if __name__ == "__main__":
    _main()
