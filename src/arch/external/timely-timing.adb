--
--  Copyright 2023 (C) Daniel King
--
--  SPDX-License-Identifier: Apache-2.0
--
package body Timely.Timing is

   function External_CYCCNT_Begin return CPU_Cycles with
     Import,
     Convention    => C,
     External_Name => "timely_cyccnt_begin";

   function External_CYCCNT_End return CPU_Cycles with
     Import,
     Convention    => C,
     External_Name => "timely_cyccnt_end";

   ------------------
   -- CYCCNT_Begin --
   ------------------

   function CYCCNT_Begin return CPU_Cycles renames External_CYCCNT_Begin;

   ----------------
   -- CYCCNT_End --
   ----------------

   function CYCCNT_End return CPU_Cycles renames External_CYCCNT_End;

end Timely.Timing;