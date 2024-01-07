--
--  Copyright 2023 (C) Daniel King
--
--  SPDX-License-Identifier: Apache-2.0
--
package body Timely.PRNG is

   procedure Split_Mix_64
     (RNG   : in out Unsigned_64;
      Value :    out Unsigned_64);

   ------------------
   -- Split_Mix_64 --
   ------------------

   procedure Split_Mix_64
     (RNG   : in out Unsigned_64;
      Value :    out Unsigned_64) is
   begin
      RNG   := RNG + 16#9E3779B97f4A7C15#;
      Value := (RNG xor Shift_Right (RNG, 30)) * 16#BF58476D1CE4E5B9#;
      Value := (Value xor Shift_Right (Value, 27)) * 16#94D049BB133111EB#;
      Value := Value xor (Shift_Right (Value, 31));
   end Split_Mix_64;

   ----------------
   -- Initialize --
   ----------------

   procedure Initialize
     (RNG  : out PRNG_State;
      Seed : Unsigned_64)
   is
      SM64_State : Unsigned_64 := Seed;
   begin
      Split_Mix_64 (SM64_State, RNG.A);
      Split_Mix_64 (SM64_State, RNG.B);
      Split_Mix_64 (SM64_State, RNG.C);
      Split_Mix_64 (SM64_State, RNG.D);
   end Initialize;

   -----------------
   -- Random_Word --
   -----------------

   procedure Random_Word
     (RNG   : in out PRNG_State;
      Value :    out Unsigned_64)
   is
      T : constant Unsigned_64 := Shift_Left (RNG.B, 17);
   begin
      Value := Rotate_Left (RNG.B * 5, 7) * 9;

      RNG.C := RNG.C xor RNG.A;
      RNG.D := RNG.D xor RNG.B;
      RNG.B := RNG.B xor RNG.C;
      RNG.A := RNG.A xor RNG.D;

      RNG.C := RNG.C xor T;
      RNG.D := Rotate_Left (RNG.D, 45);
   end Random_Word;

   --------------------
   -- Random_Natural --
   --------------------

   procedure Random_Natural
     (RNG   : in out PRNG_State;
      First :        Natural;
      Last  :        Natural;
      Value :    out Natural)
   is
      Word       : Unsigned_64;
      Num_Values : Unsigned_64;
      Limit      : Unsigned_64;

   begin
      Num_Values := Unsigned_64 (Last) - Unsigned_64 (First) + 1;

      --  Calculate the largest possible 64-bit value that is evenly divisible
      --  by Num_Values.

      Limit  := Unsigned_64'Last - (Unsigned_64'Last mod Num_Values);

      --  Generate a random word by rejection sampling. In the worst case,
      --  this has 50% chance of exiting the loop on each iteration.

      loop
         Random_Word (RNG, Word);
         exit when Word <= Limit;
      end loop;

      Value := First + Natural (Word mod Num_Values);

   end Random_Natural;

   -------------
   -- Shuffle --
   -------------

   procedure Shuffle
     (RNG : in out PRNG_State;
      Arr : in out Element_Array)
   is
      Pos  : Natural;
      Temp : Element_Type;
   begin
      for I in Natural range Arr'First .. Arr'Last - 1 loop
         PRNG.Random_Natural (RNG, I, Arr'Last, Pos);

         Temp      := Arr (I);
         Arr (I)   := Arr (Pos);
         Arr (Pos) := Temp;
      end loop;
   end Shuffle;

end Timely.PRNG;