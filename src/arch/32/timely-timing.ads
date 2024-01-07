--
--  Copyright 2023 (C) Daniel King
--
--  SPDX-License-Identifier: Apache-2.0
--
with Interfaces; use Interfaces;

package Timely.Timing is

   type CPU_Cycles is new Unsigned_32;

   function CYCCNT_Begin return CPU_Cycles;
   --  Read the architecture-specific cycle counter. This should be called
   --  at the start of the measurement period.

   function CYCCNT_End return CPU_Cycles;
   --  Read the architecture-specific cycle counter. This should be called
   --  at the end of the measurement period.

end Timely.Timing;