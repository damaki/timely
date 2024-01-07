--
--  Copyright 2023 (C) Daniel King
--
--  SPDX-License-Identifier: Apache-2.0
--
with System.Machine_Code; use System.Machine_Code;

--  This is the implementation that uses the Armv7-M DWT cycle counter

package body Timely.Timing is

   DWT_CONTROL : Unsigned_32 with
     Import,
     Address => System'To_Address (16#E000_1000#);

   DWT_CYCCNT : CPU_Cycles with
     Import,
     Address => System'To_Address (16#E000_1004#);

   LAR : Unsigned_32 with
     Import,
     Address => System'To_Address (16#E000_1FB0#);

   DEMCR : Unsigned_32 with
     Import,
     Address => System'To_Address (16#E000_EDFC#);

   LAR_Key : constant Unsigned_32 := 16#C5AC_CE55#;

   ------------------
   -- CYCCNT_Begin --
   ------------------

   function CYCCNT_Begin return CPU_Cycles is (DWT_CYCCNT);

   ----------------
   -- CYCCNT_End --
   ----------------

   function CYCCNT_End return CPU_Cycles renames CYCCNT_Begin;

begin
   DEMCR       := DEMCR or 16#0100_0000#; --  Enable trace
   LAR         := LAR_Key;                --  Unlock access to DWT
   DWT_CYCCNT  := 0;                      --  Clear DWT cycle counter
   DWT_CONTROL := DWT_CONTROL or 1;       --  Enable DWT cycle counter
end Timely.Timing;