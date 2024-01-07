--
--  Copyright 2023 (C) Daniel King
--
--  SPDX-License-Identifier: Apache-2.0
--
with System.Machine_Code; use System.Machine_Code;

--  This is the implementation that uses the Cortex-M SysTick as a cycle
--  counter.
--
--  Note that this code assumes it has exclusive access to the SysTick.
--  It is therefore not compatible with any other code that uses the SysTick,
--  such as some Ada tasking runtimes that use the SysTick to implement the
--  semantics of Ada.Real_Time.

package body Timely.Timing is

   SYST_CSR : Unsigned_32 with
     Import,
     Address => System'To_Address (16#E000_E010#);

   SYST_RVR : Unsigned_32 with
     Import,
     Address => System'To_Address (16#E000_E014#);

   SYST_CVR : Unsigned_32 with
     Import,
     Address => System'To_Address (16#E000_E018#);

   ------------------
   -- CYCCNT_Begin --
   ------------------

   function CYCCNT_Begin return CPU_Cycles is
     (CPU_Cycles (SYST_CVR and 16#FF_FFFF#));

   ----------------
   -- CYCCNT_End --
   ----------------

   function CYCCNT_End return CPU_Cycles renames CYCCNT_Begin;

begin
   SYST_CSR := SYST_CSR and 16#FFFF_FFFC#; --  ENABLE=0 and TICKINT=0
   SYST_RVR := 16#FF_FFFF#;                --  Use full 24-bit range
   SYST_CVR := 0;                          --  Any write clears to zero
   SYSR_CSR := SYST_CSR or 1;              --  ENABLE=1
end Timely.Timing;