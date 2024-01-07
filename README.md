# Timely

Timely is a library for verifying whether some code is constant time. The basic
idea is to take two or more inputs, repeatedly run the code with those inputs
and measure its execution time, then apply statistical tests to try to detect
any differences in the timings. It is based on the techniques used in
[tlsfuzzer](https://github.com/tlsfuzzer/tlsfuzzer).

This project consists of two parts:
 * an Ada test harnesses library to measure the execution time of code
   under different inputs; and
 * a Python script to run statistical tests to look for timing leakage.

## License

Apache 2.0

## Method

Timely is a dynamic analysis tool. It measures the actual execution time of the
code with different inputs on real hardware to try to detect timing leakage
via statistical methods.

While Timely is neither sound nor complete (i.e. there's the risk of false
negatives and false positives due to the statistical nature of the tests), it
measures on real hardware and therefore doesn't rely on having an accurate
hardware leakage mode and can be easily ported to different CPU architectures.

### Measurement

The execution time of the code under test is measured under two or more
different classes of input data. An input data class can be any kind of value;
it could be a fixed value that is the same for each measurement, or a value
that changes randomly for each measurement.

Input data classes are identified by a unique integer identifier
(`Timely.Class_ID`). It spans the range `0 .. N - 1` where `N` is the number of
classes to be measured. This ID is used by `Prepare_Input_Data` to set up the
input data for a class.

All execution times are measured using the processor's cycle counter. The
measurements are output in CSV format to the standard output, where the columns
contain the measurements for each input data class.

### Statistical Tests

Timely uses these statistical tests to look for timing leakage:
* [Friedman test](https://en.wikipedia.org/wiki/Friedman_test)
* [Dependent t-test for paired samples](https://en.wikipedia.org/wiki/Student%27s_t-test#Dependent_t-test_for_paired_samples)
* [Wilcoxon signed-rank test](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
* [Sign test](https://en.wikipedia.org/wiki/Sign_test)

## Usage

### Setting up a test harness

To set up a test harness, instantiate the package `Timely.Measurement` with the following:
* `Input_Data_Type`: The data type of the inputs.
* `Prepare_Input_Data`: Sets up input data for an input data class.
* `Do_One_Computation`: Runs the code whose execution time is to be measured.

Then call the `Run` procedure in the instantiated test harness to run the
harness and produce measurements.

See the `examples` directory for some example test harnesses.

### Acquiring timing measurements

To get timing measurements, simply run the program and capture its output to a file.

```sh
./example_harness_program > timings.csv
```

This will infinitely output measurements until the program is terminated.

To get a specific number of measurements, e.g. 1,000,000 measurements:

```sh
./example_harness_program | head -n 1000000 > timings.csv
```

### Checking for timing leakage

Use the `analyse.py` script to run the statistical tests and check for timing
leakage:

```sh
python analyse.py --output-dir=results timings.csv
```

See `python analyse.py --help` for details.

The script will run the statistical tests (this may take a long time) and
prints a summary of the results to the standard output. Additional reports
and graphs are output in the directory specified by `--output-dir`.

Timing leakage is considered to be present when a [_p_-value](https://en.wikipedia.org/wiki/P-value)
from any statistical test is less than the threshold value set by `--alpha`
(1e-5 by default), adjusted with the
[Bonferroni correction](https://en.wikipedia.org/wiki/Bonferroni_correction)
when applicable.

## Crate Configuration Variables

The cycle counter source to use is configured via the `timely.Arch_Timer`
crate configuration variable in your crate's `alire.toml`. This variable has
one of the following values:
* `x86_64` (default) uses the RDTSC instruction as the cycle counter source.
* `aarch64` uses the PMCCNTR_EL0 register as the cycle counter source.
* `armv7m_dwt` uses the Armv7-M DWT cycle counter.
* `cortexm_systick` uses the Cortex-M SysTick as the cycle counter source.
* `external_32` uses an externally-provided 32-bit timer as the cycle counter source.
* `external_64` uses an externally-provided 64-bit timer as the cycle counter source.

`external_32` and `external_64` use a user-defined 32-bit or 64-bit timer.
When these are selected, you must implement two functions that sample your
high resolution timer/cycle counter. The Ada signature is:
```ada
function CYCCNT_Begin return Interfaces.Unsigned_32 with
  Export,
  Convention => C,
  External_Name => "timely_cyccnt_begin";

function CYCCNT_End return Interfaces.Unsigned_32 with
  Export,
  Convention => C,
  External_Name => "timely_cyccnt_end";
```

or the equivalent C function signatures:
```c
uint32_t timely_cyccnt_begin(void);
uint32_t timely_cyccnt_end(void);
```

Timely calls these functions at the beginning and end of the measurement period.

When `external_64` is used then the return type must be `Interfaces.Unsigned_64`
for Ada or `uint64_t` for C.