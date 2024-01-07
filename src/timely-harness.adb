--
--  Copyright 2023 (C) Daniel King
--
--  SPDX-License-Identifier: Apache-2.0
--
with Ada.Unchecked_Deallocation;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.IO_Exceptions;

package body Timely.Harness is
   use Timely.Timing;

   procedure Shuffle is new PRNG.Shuffle
     (Element_Type  => Class_ID,
      Element_Array => Class_ID_Array);
   --  Randomly shuffle an array of class IDs

   procedure Generate_Random_Measurement_Sequence
     (Classes     : out Class_ID_Array;
      Num_Classes :     Positive;
      RNG         : in out PRNG.PRNG_State)
   with
     Pre => Classes'Length mod Num_Classes = 0;
   --  We measure the classes in a random order, but in pairs (or larger
   --  tuples). For example, given 3 classes and 12 measurements the sequence
   --  could be:
   --
   ---    0, 2, 1, 2, 0, 1, 1, 2, 0, 1, 0, 2
   --     |-----|  |-----|  |-----|  |-----|
   --      first   second    third   fourth
   --      tuple    tuple    tuple    tuple
   --
   --  This ensures ensures that the code executed before each measurement is
   --  random and with the same distribution for each input, which avoids
   --  environmental effects (e.g. branch prediction, caching) from affecting
   --  the timing measurements of one class more than others.

   procedure Prepare_Inputs
     (Classes    :     Class_ID_Array;
      Input_Data : out Input_Data_Array)
   with
     Pre => (Classes'First    = 1 and then
             Input_Data'First = 1 and then
             Classes'Last = Input_Data'Last);
   --  Prepare the input data for each class

   procedure Measure_Execution_Times
     (Input_Data : in out Input_Data_Array;
      Exec_Times :    out CPU_Cycles_Array)
   with
     Pre => (Exec_Times'First = 1 and then
             Input_Data'First = 1 and then
             Exec_Times'Last  = Input_Data'Last);
   --  Call Do_One_Computation over the sequence of input data and record
   --  the execution time of each computation.

   ---------
   -- Run --
   ---------

   procedure Run
     (Num_Measurements : Positive;
      Num_Classes      : Positive;
      RNG_Seed         : Unsigned_64 := Default_RNG_Seed)
   is
      procedure Free is new Ada.Unchecked_Deallocation
        (Timing_Data, Timing_Data_Access);

      Data : Timing_Data_Access := new Timing_Data (Num_Measurements);

   begin
      Initialize (Data.all, RNG_Seed);

      loop
         Measure (Data.all, Num_Classes);

         begin
            Print_Measurements_CSV (Data.all);
         exception
            --  Gracefully exit the test harness if we fail to write to the
            --  standard output. This can happen when a pipe is closed.

            when Ada.IO_Exceptions.Status_Error |
                 Ada.IO_Exceptions.Device_Error =>
               exit;
         end;
      end loop;

      Free (Data);

   exception
      when others =>
         Free (Data);
         raise;
   end Run;

   ----------------
   -- Initialize --
   ----------------

   procedure Initialize
     (Data     : out Timing_Data;
      RNG_Seed :     Unsigned_64 := Default_RNG_Seed)
   is
   begin
      Data.Num_Measurements  := 0;
      Data.Num_Classes       := 1;
      Data.Classes           := (others => 1);
      Data.Exec_Times        := (others => 0);
      PRNG.Initialize (Data.RNG, RNG_Seed);
   end Initialize;

   -------------
   -- Measure --
   -------------

   procedure Measure
     (Data        : in out Timing_Data;
      Num_Classes :        Positive)
   is
   begin
      Data.Num_Classes := Num_Classes;

      Data.Num_Measurements :=
        Data.Max_Measurements - (Data.Max_Measurements mod Num_Classes);

      Generate_Random_Measurement_Sequence
        (Classes     => Data.Classes (1 .. Data.Num_Measurements),
         Num_Classes => Data.Num_Classes,
         RNG         => Data.RNG);

      Prepare_Inputs
        (Classes    => Data.Classes (1 .. Data.Num_Measurements),
         Input_Data => Data.Input_Data (1 .. Data.Num_Measurements));

      Measure_Execution_Times
        (Input_Data => Data.Input_Data (1 .. Data.Num_Measurements),
         Exec_Times => Data.Exec_Times (1 .. Data.Num_Measurements));
   end Measure;

   ----------------------------
   -- Print_Measurements_CSV --
   ----------------------------

   procedure Print_Measurements_CSV (Data : Timing_Data) is
      type Class_Exec_Time_Array is
        array (Class_ID range 0 .. Class_ID (Data.Num_Classes - 1))
        of CPU_Cycles;

      Row : Class_Exec_Time_Array;

      Class : Class_ID;

      Pos       : Positive;
      Offset    : Natural := 0;
      Remaining : Integer := Data.Num_Measurements;
   begin
      while Remaining > 0 loop

         --  Each class tuple was measured in a random order. Gather the
         --  measurements in ascending class order so that each row of printed
         --  output is printed in the same order.

         for I in Row'Range loop
            Pos         := Data.Classes'First + Offset + Natural (I);
            Class       := Data.Classes (Pos);
            Row (Class) := Data.Exec_Times (Pos);
         end loop;

         for I in Row'Range loop
            Put (Row (I)'Image);
            if I < Row'Last then
               Put (",");
            end if;
         end loop;
         New_Line;

         Offset    := Offset    + Data.Num_Classes;
         Remaining := Remaining - Data.Num_Classes;
      end loop;
   end Print_Measurements_CSV;

   ------------------------------------------
   -- Generate_Random_Measurement_Sequence --
   ------------------------------------------

   procedure Generate_Random_Measurement_Sequence
     (Classes     : out    Class_ID_Array;
      Num_Classes :        Positive;
      RNG         : in out PRNG.PRNG_State)
   is
      Classes_Tuple : Class_ID_Array (0 .. Num_Classes - 1);
      Pos           : Positive;

   begin
      --  Get a tuple of all the class IDs to measure (1, 2, ..., Num_Classes)

      for I in Classes_Tuple'Range loop
         Classes_Tuple (I) := Class_ID (I);
      end loop;

      --  Generate a sequence of tuples to measure, where the class IDs within
      --  each tuple are randomly shuffled each time.

      for I in Natural range 0 .. (Classes'Length / Num_Classes) - 1
      loop
         Shuffle (RNG, Classes_Tuple);

         Pos := Classes'First + (I * Num_Classes);

         Classes (Pos .. Pos + Num_Classes - 1) := Classes_Tuple;
      end loop;
   end Generate_Random_Measurement_Sequence;

   --------------------
   -- Prepare_Inputs --
   --------------------

   procedure Prepare_Inputs
     (Classes    :     Class_ID_Array;
      Input_Data : out Input_Data_Array)
   is
   begin
      for I in Input_Data'Range loop
         Prepare_Input_Data (Input_Data (I), Classes (I));
      end loop;
   end Prepare_Inputs;

   -----------------------------
   -- Measure_Execution_Times --
   -----------------------------

   procedure Measure_Execution_Times
     (Input_Data : in out Input_Data_Array;
      Exec_Times :    out CPU_Cycles_Array)
   is
      Start_Time : CPU_Cycles;
      End_Time   : CPU_Cycles;

   begin
      --  This loop is as simple as possible to avoid introducing any timing
      --  side effects from the test harness, which could introduce false
      --  positive timing leakage signals.

      for I in Exec_Times'Range loop
         Start_Time := CYCCNT_Begin;
         Do_One_Computation (Input_Data (I));
         End_Time := CYCCNT_End;

         Exec_Times (I) := End_Time - Start_Time;
      end loop;
   end Measure_Execution_Times;

end Timely.Harness;
