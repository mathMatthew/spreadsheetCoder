# XML Format for SpreadsheetCoder Translations

## Overview

SpreadsheetCoder employs XML libraries to enhance functionality, offering two types:
- **Function Definitions**
- **Transformations**

Instead of altering the SpreadsheetCoder core, users can extend its functions using these libraries. 

For basic Excel functions like `+`, `-`, `*`, and others, SpreadsheetCoder provides hard-coded translations within core. Because of these (as outlined in readme), users can:
1. Craft the function in Excel.
2. Use SpreadsheetCoder to translate it the target language.

One could also use SpreadsheetCoder to write functions that use your own custom functions and even convert those functions to a different target language. The process of doing that is Extending the capabilities of SpreadsheetCoder by teaching it how to translate your own custom function. Here are the specific steps of this scenario:
1. Use SpreadsheetCoder to translate the function into VisualBasic
2. Use SpreadsheetCoder into XML (if strict mode is enabled, this will happen automatically and this separate step and the next step are not necessary.)
3. Move the XML file into the XMLFunctions library.
4. Now, re-use the function in Excel within a larger function within Excel. This works because you created a VisualBasic version of the function (in step 1).
5. Convert the larger function within Excel into the larger function.

So that's the gist at a high-level and a simple use-case for using the XML format to extend the functionatlity of SpreadsheetCoder. Now let's dive into the details. 

## Expanding SpreadsheetCoder's Functionality using the XML

### Example:

Suppose you have a complex function in Excel and you translate this function to SQL for use on Teradata. Now suppose you decide to add some additional logic. You know a bit about Teradata and you happen to know that what you want would be easier with Teradata STROK function. STRTOK extracts tokens (substrings) from a string based on defined delimiters, returning the nth specified token. For the string "apple,banana,cherry", using a comma as the delimiter and requesting the 2nd token would return "banana". STRTOK already available on Teradata but you've got your functionality sitting in Excel and you want to maintain it there for its easy of use, portability and easy of exaplanation in terms of how it works. 

Consider three ways to do this. All three will require you to let SpreadsheetCoder know that STRTOK is a valid function in the target language (teradata) by designated the XML function as a stub function (details in the `LangSpec` section below).

### Method 1: VisualBasic

Those proficient in VisualBasic can code in VisualBasic a function mirroring the desired target language function.  In this example, develop a VBA function named `STRTOK` analogous to the Teradata version. Then, convert your custom function to the target language. 

### Method 2: Auto-VisualBasic

If users can define the logic of the target language function in Excel, they can rely on SpreadsheetCoder to generate VisualBasic code. Subsequent steps mirror Method 1.

### Method 3: Utilizing Transforms

Transforms offer an alternative. When Excel has a similar yet distinct function, a transforms bridge the gap.  `STRTOK` isn't the best example, but converting `MID` to Teradata's analogous `SUBSTRING` function is straightforward. By creating a simple transform, `MID` is converted to `SUBSTRING`.

> **Note**: 
> In 'Strict' mode, the XML function format resides in a text box on the function's page. Alternatively or in addition, storing the `.XML` format in **SpreadsheetCoder/XMLFunctions** (path alterable with `XMLFunctionLibraryPath`) grants SpreadsheetCoder ongoing access. This XML collection constitutes SpreadsheetCoder's function library, while **SpreadsheetCoder/XMLTransformations** holds transformation libraries.

# XML Functions File Documentation

## Overview

SpreadsheetCoder uses XML-like files to break down intricate functions into basic, mappable elements. These definitions aid in translating embedded functions. Once stored in "SpreadsheetCoder/XMLFunctions" (location adjustable via `XMLFunctionLibraryPath` in the SC file), SpreadsheetCoder consistently accesses this function library.

These XML Function files are generated automatically by setting the target language to XML. The file How_to_do_lookups.xlsm is meant, as the name suggests, for help in understanding the special syntax around doing lookups with SpreadsheetCoder. But it is also fine as a a more generic example of how SpreadsheetCoder generates the functional logic into other languages, including XML. You may want to refer to that file to see how things work with real examples.

Like the XML Functions file, the XML Transformation files, can be generated from SpreadsheetCoder itself, if you setup the right outline of the function in Excel. For examples with how this works in the file How_to_do_transforms.xlsm.

Below is the document structure for these files.

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
- **ProcessStub**: Dictates that this specific function name, with parameters as ordered is available in the target language. 
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

Like 

### Key Differences: Outputs

Transform files necessitate defining two primary outputs in the `Outputs` section:
1. **From**: Original pattern.
2. **To**: Transformed pattern.
It can also include as many additional ouptuts as desired. each of these is considered "safe", meaning that when the transform recognizes that pattern it won't apply this transformation to any internally matching part of the transformation.

> **Note**:
> If an XML Function has a singular output, the same result can be achieved using a transform. The transformation hinges on referencing all inputs in the 'From' cell (or upstream calculations referencing the inputs) and presenting the transformed code in the 'To' cell (or upstream calculations referenceing the inputs).
