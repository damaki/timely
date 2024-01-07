--
--  Copyright 2023 (C) Daniel King
--
--  SPDX-License-Identifier: Apache-2.0
--
with Interfaces; use Interfaces;

--  A simple, fast PRNG (xoshiro256**)

package Timely.PRNG with
  Pure
is

   type PRNG_State is private;

   procedure Initialize
     (RNG  : out PRNG_State;
      Seed : Unsigned_64);
   --  Initialize the RNG

   procedure Random_Word
     (RNG   : in out PRNG_State;
      Value :    out Unsigned_64);
   --  Generate a random 64-bit word

   procedure Random_Natural
     (RNG   : in out PRNG_State;
      First :        Natural;
      Last  :        Natural;
      Value :    out Natural)
   with
     Pre  => First <= Last,
     Post => Value in First .. Last;
   --  Generate a random natural integer in the range First .. Last

   generic
      type Element_Type is private;
      type Element_Array is array (Natural range <>) of Element_Type;
   procedure Shuffle
     (RNG : in out PRNG_State;
      Arr : in out Element_Array);
   --  Randomly shuffle the contents of an array.
   --
   --  To simplify the implementation of this procedure, the Element_Array is
   --  constrained to using type Natural for the array range.

private

   type PRNG_State is record
      A : Unsigned_64;
      B : Unsigned_64;
      C : Unsigned_64;
      D : Unsigned_64;
   end record;

end Timely.PRNG;