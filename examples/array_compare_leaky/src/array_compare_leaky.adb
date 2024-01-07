--
--  Copyright 2023 (C) Daniel King
--
--  SPDX-License-Identifier: Apache-2.0
--
with Interfaces; use Interfaces;

with Timely;
with Timely.Harness;
with Timely.PRNG;

--  This test harness measures the execution time of an Ada array comparison
--  under different inputs to test if it's constant time.
--
--  The test harness implements two things:
--  1. a procedure that generates the input data for the code whose execution
--     time is to be measured (Prepare_Input_Data); and
--  2. a procedure that executes the code whose execution time is to be
--     measured (Do_One_Computation).
--
--  In this example, the array comparison is measured under two classes of
--  input data: a fixed class (all zeroes), and a random class (a random
--  element is non-zero). The data is compared against an array of all zeroes.
--
--  The exectution times are printed to the standard output in CSV format.

procedure Array_Compare_Leaky is

   ----------------------------
   -- Input Data Preparation --
   ----------------------------

   --  Here we implement a procedure called Prepare_Input_Data that sets up
   --  the data for each computation.

   type Byte_Array is array (Positive range 1 .. 5) of Unsigned_8;

   Test_Comparison : constant Byte_Array := (others => 0);

   --  A simple PRNG is used to generate values for the random data class
   RNG : Timely.PRNG.PRNG_State;

   --  This procedure is called by Timely to initialize the input data for one
   --  computation.
   procedure Prepare_Input_Data
     (Data  : out Byte_Array;
      Class :     Timely.Class_ID)
   is
      Idx : Positive;

   begin
      case Class is
         when 0 =>
            --  Fixed data class
            Data := Test_Comparison;

         when others =>
            --  Random data class

            Timely.PRNG.Random_Natural
              (RNG   => RNG,
               First => Data'First,
               Last  => Data'Last,
               Value => Idx);

            Data := Test_Comparison;
            Data (Idx) := 1;
      end case;
   end Prepare_Input_Data;

   ------------------------
   -- Do_One_Computation --
   ------------------------

   --  This is the procedure whose execution time will be measured. It performs
   --  the array comparison over some pre-prepared data.

   procedure Do_One_Computation (Data : in out Byte_Array) is
      Result : Boolean;
   begin

      Result := Data = Test_Comparison;

      --  Prevent the compiler from optimizing away the result of the array
      --  comparison, since the assignment to 'Result' is a dead store.

      pragma Inspection_Point (Result);
   end Do_One_Computation;

   ------------------------
   -- Test Harness Setup --
   ------------------------

   --  Now we instantiate our test harness

   package Harness is new Timely.Harness
     (Input_Data_Type    => Byte_Array,
      Prepare_Input_Data => Prepare_Input_Data,
      Do_One_Computation => Do_One_Computation);

begin
   Timely.PRNG.Initialize (RNG, Seed => 123);

   Harness.Run
     (Num_Measurements => 1_000_000,
      Num_Classes      => 2);
end Array_Compare_Leaky;
