--
--  Copyright 2023 (C) Daniel King
--
--  SPDX-License-Identifier: Apache-2.0
--
with Interfaces; use Interfaces;

private with Timely.Timing;
private with Timely.PRNG;

generic
   type Input_Data_Type is limited private;

   with procedure Prepare_Input_Data
     (Data  : out Input_Data_Type;
      Class :     Class_ID);
   --  Prepare the input data for a particular class

   with procedure Do_One_Computation (Data : in out Input_Data_Type);
   --  Do a computation whose execution time is to be measured

package Timely.Harness is

   Default_RNG_Seed : constant Unsigned_64 := 123;
   --  Default value for the RNG seed, chosen arbitrarily.

   procedure Run
     (Num_Measurements : Positive;
      Num_Classes      : Positive;
      RNG_Seed         : Unsigned_64 := Default_RNG_Seed);
   --  Run the test harness.
   --
   --  This takes measurements and prints the results in an infinite loop.
   --
   --  Num_Measurements is the number of measurements to take in one go.
   --  Larger is better, but requires more memory.
   --
   --  Num_Classes is the number of input data classes to measure.
   --  Prepare_Input_Data will be called for classes in the range
   --  0 .. Num_Classes - 1.
   --
   --  RNG_Seed seeds the internal PRNG used to randomize the measurement
   --  sequence. The exact value doesn't matter, different values will cause
   --  different measurement sequences.

   --------------------------
   -- Additional Functions --
   --------------------------

   type Timing_Data (Max_Measurements : Positive) is limited private;

   type Timing_Data_Access is access Timing_Data;

   procedure Initialize
     (Data     : out Timing_Data;
      RNG_Seed :     Unsigned_64 := Default_RNG_Seed);
   --  Initialize the harness.
   --
   --  RNG_Seed seeds the internal PRNG used to randomize the measurement
   --  sequence. The exact value doesn't matter, different values will cause
   --  different measurement sequences.

   procedure Measure
     (Data        : in out Timing_Data;
      Num_Classes :        Positive)
   with
     Pre => Num_Classes <= Data.Max_Measurements;
   --  Gather timing measurements.

   procedure Print_Measurements_CSV (Data : Timing_Data);
   --  Print the timing measurements to the standard output in CSV format.
   --
   --  Each column in the CSV data contains the execution times for one
   --  class of input data. The first column contains the measurements
   --  for class 0, the second column for class 1, and so on...

private

   type Class_ID_Array     is array (Natural range <>)  of Class_ID;
   type CPU_Cycles_Array   is array (Positive range <>) of Timing.CPU_Cycles;
   type Input_Data_Array   is array (Positive range <>) of Input_Data_Type;

   type Timing_Data (Max_Measurements : Positive) is limited
   record
      RNG : PRNG.PRNG_State;
      --  Used to randomise the order of the measurements

      Num_Measurements : Natural;
      --  Records the number of measurements. Does not exceed Max_Measurements

      Num_Classes : Positive;
      --  Number of classes being measured

      Input_Data : Input_Data_Array   (1 .. Max_Measurements);
      --  Stores the input data for each measurement

      Exec_Times : CPU_Cycles_Array   (1 .. Max_Measurements);
      --  Records the execution time of each measurement

      Classes : Class_ID_Array (1 .. Max_Measurements);
      --  Records the class number of each measurement

   end record with
     Type_Invariant => (Num_Measurements <= Max_Measurements and then
                        Num_Classes      <= Max_Measurements and then
                        Num_Measurements mod Num_Classes = 0);

end Timely.Harness;
