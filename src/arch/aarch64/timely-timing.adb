--
--  Copyright 2023 (C) Daniel King
--
--  SPDX-License-Identifier: Apache-2.0
--
with System.Machine_Code; use System.Machine_Code;

package body Timely.Timing is
   use ASCII;

   ------------------
   -- CYCCNT_Begin --
   ------------------

   function CYCCNT_Begin return CPU_Cycles is
      Result : Unsigned_64;
   begin
      Asm ("isb" & LF &
           "mrs %0, PMCCNTR_EL0",
           Outputs => (Unsigned_64'Asm_Output ("=r", Result)),
           Volatile => True);

      return CPU_Cycles (Result);
   end CYCCNT_Begin;

   ----------------
   -- CYCCNT_End --
   ----------------

   function CYCCNT_End return CPU_Cycles is
      Result : Unsigned_64;
   begin
      Asm ("isb" & LF &
           "mrs %0, PMCCNTR_EL0",
           Outputs => (Unsigned_64'Asm_Output ("=r", Result)),
           Volatile => True);

      return CPU_Cycles (Result);
   end CYCCNT_End;

end Timely.Timing;