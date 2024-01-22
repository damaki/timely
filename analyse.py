import argparse
import csv
import concurrent.futures
import enlighten
import multiprocessing.shared_memory
import matplotlib
import numpy as np
import pandas as pd
import scipy.stats
import sys
import tempfile
import threading
from contextlib import contextmanager
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Callable, Dict, List, Tuple, Optional, Any
from pathlib import Path


class BinarySamplesFileLoader:
    """Loads binary samples as from memory-mapped file"""

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

    Using a binary file format instead of CSV allows the samples to be memory
    mapped into multiple worker processes and therefore avoid excessive memory
    usage when dealing with large sample sets and many CPU cores.

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
    with pd.read_csv(
        samples_csv_path, chunksize=512000, dtype=dtype, header=None
    ) as csv_reader:
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
        ax.set_ylabel("Cycles")
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
        ax.set_ylabel("Cycles")
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
        ax.set_ylabel("Cycles")
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


def _resample(samples):
    """Create a resample of the provided samples with the same length."""
    return np.random.choice(samples, replace=True, size=len(samples))


def _trimean(quartile1, median, quartile3):
    """Calculate the statistical trimean."""
    return (quartile1 + (median * 2) + quartile3) / 4


def _central_tendencies_of_random_samples(
    shared_mem: multiprocessing.shared_memory.SharedMemory,
    shape,
    dtype: np.dtype,
    num_resamples: int,
) -> Dict[str, List[float]]:
    """Calculate the central tendencies of multiple resamplings of some data.

    This function is intended to be called in a subprocess, so the samples are
    passed in a shared memory buffer for efficiency.

    Parameters
    ----------
    shared_mem
        The shared memory buffer containing the samples to process.
    shape
        The shape of the numpy array in the shared_mem buffer.
    dtype
        The dtype of the numpy array elements in the shared_mem buffer.
    num_resamples
        The number of resamplings to perform.

    Returns
    -------
    The dict which maps a central tendency name to the list of values (one
    value per resampling).
    """

    samples = np.ndarray(shape, dtype, buffer=shared_mem.buf)

    means = []
    medians = []
    trimmed_means_5pc = []
    trimmed_means_25pc = []
    trimmed_means_45pc = []
    trimeans = []

    for _ in range(num_resamples):
        resampled = _resample(samples)

        q1, median, q3 = np.quantile(resampled, [0.25, 0.5, 0.75])

        means.append(np.mean(resampled))
        medians.append(median)
        trimmed_means_5pc.append(scipy.stats.trim_mean(resampled, 0.05))
        trimmed_means_25pc.append(scipy.stats.trim_mean(resampled, 0.25))
        trimmed_means_45pc.append(scipy.stats.trim_mean(resampled, 0.45))
        trimeans.append(_trimean(q1, median, q3))

    return {
        "mean": means,
        "median": medians,
        "trimmed_mean_5pc": trimmed_means_5pc,
        "trimmed_mean_25pc": trimmed_means_25pc,
        "trimmed_mean_45pc": trimmed_means_45pc,
        "trimean": trimeans,
    }


def _bootstrap_central_tendencies_of_differences(
    samples: np.ndarray,
    class1: int,
    class2: int,
    num_resamples: int,
    submit_job: Callable[[Callable, Any], concurrent.futures.Future],
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """Calculate the central tendencies of pairwise differences via bootstrapping.

    This operation can take a long time, so the work is distributed over a
    process pool to take advantage of multiple cores.

    Parameters
    ----------
    samples
        The timing samples for all input data classes
    class1
        The index of the first class to compare
    class2
        The index of the second class to compare
    num_resamples
        The number of resamplings to perform.
    submit_job
        Callable used to submit jobs to a process pool

    Returns
    -------
    exact_central_tendencies
        The central tendencies over the pairwise differences
    bootstrapped_central_tendencies
        The central tendencies for multiple resamplings of the pairwise differences
    """

    # Large values of 'num_resamples' would take a very long time to compute
    # on a single thread, so we spread the load over multiple parallel workers
    # to speed things up.
    #
    # Shared memory is to send the pairwise differences to the worker
    # processes since this is much more efficient than pickling.

    nsamples = samples.shape[0]
    differences_buffer_size = nsamples * samples.itemsize
    differences_mem = multiprocessing.shared_memory.SharedMemory(
        create=True, size=differences_buffer_size
    )

    try:
        differences = np.ndarray((nsamples,), samples.dtype, buffer=differences_mem.buf)
        differences[:] = samples[:, class2] - samples[:, class1]

        # Split the work into multiple parallel jobs. For data sets larger than
        # 10 million the number of resamplings per job is limited to one resampling
        # so that jobs finish quickly enough to regularly report progress.
        # Otherwise, we scale the job to allow more work per job while still
        # keeping a reasonable limit on the job duration.
        if nsamples >= 1e7:
            resamples_per_job = 1
        else:
            resamples_per_job = int(min(1e7 // nsamples, num_resamples))

        job_sizes = [resamples_per_job] * (num_resamples // resamples_per_job)

        last_partial_job_size = num_resamples % resamples_per_job
        if last_partial_job_size > 0:
            job_sizes.append(last_partial_job_size)

        # Submit the jobs
        futures = [
            submit_job(
                _central_tendencies_of_random_samples,
                differences_mem,
                differences.shape,
                differences.dtype,
                n,
            )
            for n in job_sizes
        ]

        # Compute the exact (non-resampled) central tendencies in parallel with
        # the bootstrapping computation
        q1, median, q3 = np.quantile(differences, [0.25, 0.5, 0.75])
        exact_central_tendencies = {
            "mean": np.mean(differences),
            "median": median,
            "trimmed_mean_5pc": scipy.stats.trim_mean(differences, 0.05),
            "trimmed_mean_25pc": scipy.stats.trim_mean(differences, 0.25),
            "trimmed_mean_45pc": scipy.stats.trim_mean(differences, 0.45),
            "trimean": _trimean(q1, median, q3),
        }

        # Rejoin the results from each job back into a single dictionary
        bootstrapped_central_tendencies = {}
        for future in concurrent.futures.as_completed(futures):
            job_result = future.result()

            for key, values in job_result.items():
                if key not in bootstrapped_central_tendencies:
                    bootstrapped_central_tendencies[key] = values
                else:
                    bootstrapped_central_tendencies[key].extend(values)

        return exact_central_tendencies, bootstrapped_central_tendencies
    finally:
        del differences
        differences_mem.close()
        differences_mem.unlink()


def _confidence_interval_of_central_tendencies_of_differences(
    samples_bin: BinarySamplesFileLoader,
    class1: int,
    class2: int,
    num_resamples: int,
    confidence_level: float,
    submit_job: Callable[[Callable, Any], concurrent.futures.Future],
) -> Dict[str, Tuple[float, float, float]]:
    """Calculate the confidence interval of central tendencies of pairwise
    differences via bootstrapping.

    Parameters
    ----------
    samples
        The samples for all classes.
    class1
        The index of the first class to compare in 'samples'.
    class2
        The index of the second class to compare in 'samples'.
    num_resamples
        The number of resamplings to perform for bootstrapping.
    confidence_level
        The confidence level to use (e.g. 0.95 for 95% confidence).
    submit_job
        Callable used to submit jobs to a process pool

    Returns
    -------
    The confidence intervals for each of the central tendencies. Each confidence
    interval is a tuple of the CI lower bound, exact value, and CI upper bound.
    """

    with samples_bin.load_samples() as samples:
        (
            exact_central_tendencies,
            bootstrapped_central_tendencies,
        ) = _bootstrap_central_tendencies_of_differences(
            samples=samples,
            class1=class1,
            class2=class2,
            num_resamples=num_resamples,
            submit_job=submit_job,
        )

    # Calculate the confidence interval for each of the central tendency statistics
    confidence_intervals = {}
    for key, value in exact_central_tendencies.items():
        lower = (1 - confidence_level) / 2
        upper = 1 - (1 - confidence_level) / 2
        quantiles = np.quantile(bootstrapped_central_tendencies[key], [lower, upper])

        confidence_intervals[key] = (quantiles[0], value, quantiles[1])

    return confidence_intervals, bootstrapped_central_tendencies


def _draw_confidence_interval_plots(
    output_dir: Path,
    confidence_intervals_futures: Dict[Tuple[int, int], concurrent.futures.Future],
):
    """Generate the confidence interval plots

    This also outputs the bootstrapped central tendencies data in CSV format.

    Parameters
    ----------
    output_dir
        The directory to where the outputs are generated
    confidence_interval_futures
        Contains the futures for the bootstrapped confidence interval interval data.
        The keys are a tuple of two class indices and the values are the futures
        for the CI computations for that class pair.
    """

    # Gather the data for each plot.
    # The keys are the name of the central tendency e.g. "mean", "median", etc.
    # The values are a DataFrame that contains the bootstrapped data for each class pair.
    plots = {}
    for (c1, c2), future in confidence_intervals_futures.items():

        # Get return value of _confidence_interval_of_central_tendencies_of_differences
        _, central_tendencies = future.result()

        for name, data in central_tendencies.items():
            if name not in plots:
                plots[name] = pd.DataFrame()

            plots[name][f"{c1} vs {c2}"] = data

    # Generate a plot for each central tendency
    for name, data in plots.items():
        fig = Figure(figsize=(16, 12))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)
        ax.violinplot(data, widths=0.7, showmeans=True, showextrema=True)
        ax.set_xticks(list(range(len(data.columns) + 1)))
        ax.set_xticklabels([" "] + list(data.columns))

        if name == "trimmed_mean_5pc":
            pretty_name = "trimmed mean (5%)"
        elif name == "trimmed_mean_25pc":
            pretty_name = "trimmed mean (25%)"
        elif name == "trimmed_mean_45pc":
            pretty_name = "trimmed mean (45%)"
        else:
            pretty_name = name

        ax.set_title(f"Confidence intervals for {pretty_name} of pairwise differences")
        ax.set_xlabel("Class pairs")
        ax.set_ylabel(f"{pretty_name} of pairwise differences\n(cycles)")

        canvas.print_figure(
            output_dir / f"conf_interval_plot_{name}.png",
            bbox_inches="tight",
        )

        with open(output_dir / f"bootstrapped_{name}.csv", "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(data.columns)
            writer.writerows(data.itertuples(index=False))


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
    confidence_level: float,
    confidence_intervals_futures: Dict[Tuple[int, int], concurrent.futures.Future],
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
    worst_pair, worst_results = _worst_pair(class_pair_futures)

    if friedman_future is not None:
        print(f"Friedman test p-value: {friedman_future.result().pvalue}")

    print(f"Worst pair: class {worst_pair[0]} and class {worst_pair[1]}")
    print(f"\tWilcoxon signed-rank test p-value: {worst_results['wilcoxon'].pvalue}")
    print(f"\tt-test p-value: {worst_results['ttest'].pvalue}")
    print(f"\tsign test p-value: {worst_results['sign_test'].pvalue}")

    if worst_pair in confidence_intervals_futures:
        confidence_intervals, _ = confidence_intervals_futures[worst_pair].result()
        max_ci = max(
            (confidence_intervals[key][2] - confidence_intervals[key][0]) / 2
            for key in [
                "mean",
                "trimmed_mean_5pc",
                "trimmed_mean_25pc",
                "trimmed_mean_45pc",
            ]
        )
    else:
        max_ci = None

    has_leakage = _timing_leakage_detected(friedman_future, class_pair_futures, alpha)
    print()
    if has_leakage:
        print("Result: Timing leakage DETECTED. Implementation is VULNERABLE.")
    elif max_ci is None:
        print(
            "Result: No timing leakage was found, "
            "but cannot be certain that the code is free from timing side "
            "channels because the confidence interval calculation was skipped."
        )
    elif max_ci < 1.0:
        # The code is considered to be constant time when the
        print(
            f"Result: No timing leakage was found. The confidence interval suggests "
            f"that the code is free from timing side channels "
            f"(side channel signal is less than {max_ci:.3f} cycles with "
            f"{confidence_level * 100.0:.1f}% confidence)."
        )
    else:
        # Side channel signal is more than 1 cycle, so timing leakage could be
        # possible.
        print(
            f"Result: No timing leakage was found, but the confidence interval suggests "
            f"that timing leakage up to {max_ci:.3f} cycles could be possible."
        )
        print("More data is needed to confirm that the code is constant time.")

    return has_leakage


def analyse(
    samples_bin: BinarySamplesFileLoader,
    output_dir: Optional[Path],
    alpha: float = 1e-5,
    confidence_level: Optional[float] = 0.95,
    num_resamples: int = 5000,
    skip_friedman: bool = False,
    ci_plot_all: bool = False,
    max_workers: Optional[int] = None,
):
    """Run all statistical tests and generate various output plots

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
    confidence_level
        Confidence level to use. If this is None then the confidence interval
        calculations are skipped.
    num_resamples
        Number of resamplings to use when bootstrapping the confidence interval.
    skip_friedman
        When True, don't run the Friedman test.
    ci_plot_all:
        When True, a confidence interval will be bootstrapped for all pairs of
        input data classes. Note that this will *significantly* increase the
        analysis time.
    max_workers:
        Maximum number of worker subprocesses to use. If None, this will use
        all available cores on the machine.

    Returns
    -------
    True if evidence of timing leakage was found, or False otherwise.
    """
    nsamples = samples_bin.nsamples
    nclasses = samples_bin.nclasses

    # The analysis can take a very long time, so use progress bars to
    # show some feedback to the user.
    manager = enlighten.get_manager()
    manager.no_resize = True
    stats_progbar = manager.counter(total=0, desc="Statistical tests")
    plots_progbar = manager.counter(total=0, desc="Generating Plots ")
    if confidence_level is not None:
        ci_progbar = manager.counter(total=0, desc="Conf. Intervals  ")
        boots_progbar = manager.counter(total=0, desc="Bootstrapping CI ")

    # Protects the progress bars when updating from multiple threads
    progbar_lock = threading.Lock()

    # Some analyses can take a considerable amount of time, so we run them
    # in parallel to take advantage of multiple cores. Separate processes
    # are used to avoid contention with the Python GIL.

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as proc_pool:

        def submit_proc_pool_job(progbar, fn, *args, **kwargs):
            """Submit a job to the process pool and advance a progress bar
            when the job is done.
            """
            future = proc_pool.submit(fn, *args, **kwargs)
            with progbar_lock:
                progbar.total += 1
                progbar.refresh()
            future.add_done_callback(lambda _: progbar.update())
            return future

        # A thread pool is used for most things that need to wait on results
        # (futures) from the process pool since process pool jobs can't
        # block on futures without the risk of deadlocks.

        with concurrent.futures.ThreadPoolExecutor() as thread_pool:

            def submit_thread_pool_job(progbar, fn, *args, **kwargs):
                """Submit a job to the thread pool and advance a progress bar
                when the job is done.
                """
                future = thread_pool.submit(fn, *args, **kwargs)
                with progbar_lock:
                    progbar.total += 1
                    progbar.refresh()
                future.add_done_callback(lambda _: progbar.update())
                return future

            # scipy's Friedman test uses the chi square approximation which
            # requires at least 3 sets of samples and more than 10 samples in
            # each set, so we skip this test if these conditions aren't met.
            if (not skip_friedman) and nsamples > 10 and nclasses >= 3:
                friedman_future = submit_proc_pool_job(
                    stats_progbar, _friedman_test, samples_bin
                )
            else:
                friedman_future = None

            # Compare all pairs of classes
            class_pair_futures = {}
            for c1 in range(nclasses):
                for c2 in range(c1 + 1, nclasses):
                    class_pair_futures[(c1, c2)] = submit_proc_pool_job(
                        stats_progbar, _class_pair_statistics, samples_bin, c1, c2
                    )

            submit_proc_pool_job(
                plots_progbar,
                _draw_samples_scatter_plot,
                output_dir / "scatter_plot.png",
                samples_bin,
            )
            submit_proc_pool_job(
                plots_progbar,
                _draw_classes_box_plot,
                output_dir / "box_plot.png",
                samples_bin,
            )
            submit_proc_pool_job(
                plots_progbar,
                _draw_class_means_plot,
                output_dir / "class_means.png",
                samples_bin,
            )

            thread_pool.submit(
                _generate_class_pair_csv_report,
                output_dir / "class_pair_stats.csv",
                class_pair_futures,
            )

            # Determine which class pairs to bootstrap a confidence interval.
            ci_class_pairs = []
            if confidence_level is not None:
                if ci_plot_all:
                    for c1 in range(nclasses):
                        for c2 in range(c1 + 1, nclasses):
                            ci_class_pairs.append((c1, c2))
                else:
                    worst_pair, _ = _worst_pair(class_pair_futures)
                    ci_class_pairs.append(worst_pair)

            # Submit the jobs for the confidence intervals
            confidence_intervals_futures = {}
            for c1, c2 in ci_class_pairs:
                confidence_intervals_futures[(c1, c2)] = submit_thread_pool_job(
                    ci_progbar,
                    _confidence_interval_of_central_tendencies_of_differences,
                    samples_bin=samples_bin,
                    class1=c1,
                    class2=c2,
                    num_resamples=num_resamples,
                    confidence_level=confidence_level,
                    submit_job=lambda fn, *args: submit_proc_pool_job(
                        boots_progbar, fn, *args
                    ),
                )

            if len(ci_class_pairs) > 0:
                submit_thread_pool_job(
                    plots_progbar,
                    _draw_confidence_interval_plots,
                    output_dir=output_dir,
                    confidence_intervals_futures=confidence_intervals_futures,
                )

            # Wait for everything to finish
            thread_pool.shutdown(wait=True)
            proc_pool.shutdown(wait=True)

            has_leakage = _print_summary(
                class_pair_futures,
                friedman_future,
                alpha,
                confidence_level,
                confidence_intervals_futures,
            )

    manager.stop()
    return has_leakage


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
        "-c",
        "--conf-level",
        nargs="?",
        type=float,
        default=0.95,
        help=(
            "Confidence level to use for the confidence interval. "
            "E.g. 0.95 means 95% confidence."
        ),
    )
    parser.add_argument(
        "--ci-all-classes",
        action="store_true",
        default=False,
        help=(
            "Calculate the confidence interval for all class pairs "
            "(warning: *significantly* increases analysis time)"
        ),
    )
    parser.add_argument(
        "--no-ci",
        action="store_true",
        default=False,
        help="Skip confidence interval calculation"
    )
    parser.add_argument(
        "-r",
        "--resamples",
        nargs="?",
        default=5000,
        help="Number of resamplings to use for bootstrapping the confidence interval",
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
            ci_plot_all=args.ci_all_classes and not args.no_ci,
            alpha=args.alpha,
            confidence_level=args.conf_level if not args.no_ci else None,
            num_resamples=args.resamples,
            max_workers=args.jobs,
        )

        if has_leakage:
            sys.exit(-1)


if __name__ == "__main__":
    _main()
