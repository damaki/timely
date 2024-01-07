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
      L, H : Unsigned_32;
   begin
      --  "If software requires RDTSC to be executed only after all previous
      --  instructions have executed and all previous loads and stores are
      --  globally visible, it can execute the sequence MFENCE;LFENCE
      --  immediately before RDTSC."

      Asm ("mfence" & LF &
           "lfence" & LF &
           "rdtsc",
           Outputs => (Unsigned_32'Asm_Output ("=a", L),
                       Unsigned_32'Asm_Output ("=d", H)),
           Volatile => True);

      return CPU_Cycles (Shift_Left (Unsigned_64 (H), 32) or Unsigned_64 (L));
   end CYCCNT_Begin;

   ----------------
   -- CYCCNT_End --
   ----------------

   function CYCCNT_End return CPU_Cycles is
      L, H : Unsigned_32;
   begin
      --  "If software requires RDTSC to be executed prior to execution of any
      --  subsequent instruction (including any memory accesses), it can
      --  execute the sequence LFENCE immediately after RDTSC."

      Asm ("rdtsc" & LF &
           "lfence",
           Outputs => (Unsigned_32'Asm_Output ("=a", L),
                       Unsigned_32'Asm_Output ("=d", H)),
           Volatile => True);

      return CPU_Cycles (Shift_Left (Unsigned_64 (H), 32) or Unsigned_64 (L));
   end CYCCNT_End;

end Timely.Timing;