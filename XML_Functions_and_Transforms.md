# XML Format for SpreadsheetCoder Translations

## Overview

SpreadsheetCoder employs XML libraries to enhance flexibility, offering two types:
- **Function Definitions**
- **Transformations**

Instead of altering the SpreadsheetCoder core, users can extend its functions using these libraries. 

For basic Excel functions like `+`, `-`, `*`, and others, SpreadsheetCoder provides hard-coded translations. Users can:
1. Craft the function in Excel.
2. Use SpreadsheetCoder to translate it into VisualBasic for internal Excel utilization.
3. Convert it into the chosen target language.

Note: If the user uses the VisualBasic version of certain set of logical steps to create a larger Excel function, SpreadsheetCoder will utilize the definition of the component function to translate the larger Excel function.

For functionalities outside SpreadsheetCoder's default offerings, users can:
1. Adjust the open-source SpreadsheetCoder code.
2. Implement the XML libraries without modifying the core.

## Expanding SpreadsheetCoder's Functionality using the XML

### Example:

You wish to employ the Teradata `STRTOK` function within a custom Excel function you will create. STRTOK extracts tokens (substrings) from a string based on defined delimiters, returning the nth specified token. For the string "apple,banana,cherry", using a comma as the delimiter and requesting the 2nd token would return "banana". Consider three ways to do this. All three will require you to let SpreadsheetCoder know that STRTOK is a valid function in the target language (teradata) by designated the XML function as a stub function (details in the `LangSpec` section below).

### Method 1: VisualBasic

For those proficient in VisualBasic,  code in VisualBasic a function mirroring the desired target language function.  In this example, develop a VBA function named `STRTOK` analogous to the Teradata version. Then, convert your custom function to the target language.

### Method 2: Auto-VisualBasic

If users can define the logic of the target language function in Excel, they can rely on SpreadsheetCoder to generate VisualBasic code. Subsequent steps mirror Method 1.

### Method 3: Utilizing Transforms

Transforms offer an alternative. When Excel has a similar yet distinct function, transforms bridge the gap.  `STRTOK` isn't the best example, but converting `MID` to Teradata's analogous `SUBSTRING` function is straightforward. By creating a simple transform, `MID` is converted to `SUBSTRING`.

> **Note**: 
> In 'Strict' mode, the XML function format resides in a text box on the function's page. Alternatively or in addition, storing the `.XML` format in **SpreadsheetCoder/XMLFunctions** (path alterable with `XMLFunctionLibraryPath`) grants SpreadsheetCoder ongoing access. This XML collection constitutes SpreadsheetCoder's function library, while **SpreadsheetCoder/XMLTransformations** holds transformation libraries.

# XML Functions File Documentation

## Overview

SpreadsheetCoder uses XML-like files to break down intricate functions into basic, mappable elements. These definitions aid in translating embedded functions. Once stored in "SpreadsheetCoder/XMLFunctions" (location adjustable via `XMLFunctionLibraryPath` in the SC file), SpreadsheetCoder consistently accesses this function library.

## Key Elements:

### CodeCalculation

Root element with attributes like:
- **Name**: Defined complex function.
- **Version**: Function version.
- **HasMultipleOutputs**: Indicates if multiple outputs exist (1 for true, 0 for false).

### LangSpec

Details on target language-specific function handling. Attributes encompass:
- **Language**: Target language ID.
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
- **ProcessStub**: Dictates function processing for a particular language.
- **Skip**: Dictates if the function is processed or skipped (1 for skip, 0 for process).

### Inputs

Houses individual function input elements.
- **InputId**: The position or order of the input.
- **Name**: Descriptive name of the input.
- **Type**: The type of input. Mapped as:
  - **0**: Text
  - **1**: Number
  - **2**: Boolean
  - **3**: Date

### InputDependencies

Specifies dependencies between inputs and nodes.

### FunctionNodes

Holds each functional node definition.

### ConstantNodes

Describes constants within the function.

### NamedNodes & NodeComments

Seemingly placeholders; comprehensive documentation requires additional details.

### Outputs

Portrays function outputs.

### NodeDependencies

Illustrates interconnected nodes.

# Transform File Documentation for SpreadsheetCoder

## Overview

Transform files instruct SpreadsheetCoder to transition one format to another. The main distinction between XML Function and Transform lies in their handling of complex scenarios.

### Key Differences: Outputs

Transform files necessitate defining two primary outputs in the `Outputs` section:
1. **From**: Original pattern.
2. **To**: Transformed pattern.
It can also include as many additional ouptuts as desired. each of these is considered "safe", meaning that when the transform recognizes that pattern it won't apply this transformation to any internally matching part of the transformation.

> **Note**:
> If an XML Function has a singular output, the same result can be achieved using a transform. The transformation hinges on referencing all inputs in the 'From' cell (or upstream calculations referencing the inputs) and presenting the transformed code in the 'To' cell (or upstream calculations referenceing the inputs).
