--
--  Copyright 2023 (C) Daniel King
--
--  SPDX-License-Identifier: Apache-2.0
--
package Timely with
  Pure
is

   type Class_ID is new Natural;
   --  Identifies a unique input data class.
   --
   --  The class IDs span the range 0 .. N - 1 where N is the number of classes
   --  to be tested.
   --
   --  Leakage detection takes a set of measurements for two (or more)
   --  different classes of input data. One example is a "fix-vs-random"
   --  test where the first input data class is a constant value, and the
   --  second input data class is chosen at random for each measurement.
   --
   --  The measurement is not limited to two classes; an arbitrary number of
   --  classes can be measured and compared against each other for leakage.

end Timely;
