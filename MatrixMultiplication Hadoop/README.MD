 Matrix Multiplication using Hadoop MapReduce

## Overview
This project implements matrix multiplication using Hadoop MapReduce. It includes custom Writable classes to handle matrix elements and coordinates.


## How to Run

### Prerequisites
- Apache Hadoop
- Java Development Kit (JDK)

### Steps
1. **Compile the Java code:**
   ```bash
   javac -classpath `hadoop classpath` -d . Multiply.java
   jar -cvf multiply.jar -C . .
   ```
2. Run the Hadoop jobs:
  ```bash
  hadoop jar multiply.jar Multiply <input_path_matrix1> <input_path_matrix2> <intermediate_output_path> <final_output_path>
  ```

