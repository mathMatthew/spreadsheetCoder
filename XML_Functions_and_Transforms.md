# XML Functions File Documentation

## Overview
The SpreadsheetCoder application uses XML-like files to decompose complex functions into simpler, mappable components. These files assist in translating spreadsheet functions into various computer languages by breaking them down into basic building blocks.

## Main Elements:
### CodeCalculation
This is the root element. It contains attributes such as:
- **Name**: The name of the complex function being defined.
- **Version**: The version of the function definition.
- **HasMultipleOutputs**: A flag indicating whether the function has multiple outputs (1 for true, 0 for false).

### LangSpec
Provides details on how the function should be treated for a specific target language. Attributes include:
- **Language**: The ID representing the target language. Mapped as:
  - **1**: VB - Simple
  - **2**: SQL - Simple
  - **3**: VB - Complex
  - **4**: JS - Simple
  - **5**: XML
  - **6**: SQL - 2K
  - **7**: SQL - Tera - Script
  - **8**: SQL - Tera - Function
  - **9**: SQL - Tera - Proc
  - **10**: Excel
- **ProcessStub**: A flag indicating the approach towards processing the function for the specific language.
  - If set to **1** (true), the application will use only the name of the function and its inputs, without breaking down the function into its constituent parts. This is particularly useful when the complex function already has a direct representation in the target language.
  - If set to **0** (false), the function will be broken down as per its detailed definition.
- **Skip**: Indicates whether the function should be skipped or processed for the specified language. (1 for skip, 0 for process)

### Inputs
Contains child elements for each input the function takes.

#### Input
Represents a single input. Attributes include:
- **InputId**: The position or order of the input.
- **Name**: Descriptive name of the input.
- **Type**: The type of input. Mapped as:
  - **0**: Text
  - **1**: Number
  - **2**: Boolean
  - **3**: Date

### InputDependencies
Lists the dependencies between inputs and nodes. Each InputDependency element has:
- **InputId**: Which input it refers to.
- **NodeId**: The node that this input affects.

### FunctionNodes
Contains definitions for each functional node used in the calculation.

#### FunctionNode
Represents a specific function. Attributes include:
- **NodeId**: The ID of the function node.
- **Name**: The name of the function.
- **HasMultipleChildren**: Indicates if the function has multiple children nodes (1 for true, 0 for false).

### ConstantNodes
Defines constants used in the function.

#### ConstantNode
A specific constant value. Attributes are:
- **NodeId**: The ID of the constant node.
- **Type**: The type of constant. (Mapping is the same as for input types.)
- **Value**: The actual value of the constant.
- **HasMultipleChildren**: Indicates if the constant node has multiple children nodes (1 for true, 0 for false).

### NamedNodes and NodeComments
These sections seem to be placeholders and don't contain information in the given example. Further details would be needed to document them.

### Outputs
Describes the outputs of the function.

#### Output
A single output. Attributes include:
- **Id**: The output's ID.
- **Name**: The name of the output.
- **Type**: The type of output (Mapped as described in the Inputs section).
- **NodeId**: Refers to the node that computes this output.

### NodeDependencies
Lists the dependencies between nodes, describing how they are interconnected.

#### NodeDependency
Describes a specific connection. Attributes include:
- **ParentNodeId**: The ID of the parent node.
- **ChildNodeId**: The ID of the child node that depends on the parent.
- **ParentPosition**: Indicates the position/order in which the child node references the parent.
