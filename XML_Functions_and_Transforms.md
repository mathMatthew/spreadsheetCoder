# XML Format for spreadsheetCoder Translations

## Overview

spreadsheetCoder employs XML libraries to enhance functionality, offering two types:
- **Function Definitions**
- **Transformations**

Instead of altering the spreadsheetCoder core, users can extend its functions using these libraries. 

For basic Excel functions like `+`, `-`, `*`, and others, spreadsheetCoder provides hard-coded translations within core. Because of these (as outlined in readme), users can:
1. Craft the function in Excel.
2. Use spreadsheetCoder to translate it to the target language.

One could also use spreadsheetCoder to write functions that use your own custom functions and even convert those functions to a different target language. The process of doing that is Extending the capabilities of spreadsheetCoder by teaching it how to translate your own custom function. Here are the specific steps of this scenario:
1. Use spreadsheetCoder to translate the function into VisualBasic for Applications (VBA) code and have it add it to your spreadsheet
2. Use spreadsheetCoder to translate your function into XML (if strict mode is enabled, this will happen automatically and this separate step and the next step are not necessary.)
3. Move the XML file into the XMLFunctions library.
4. Now, use the function you added to Excel in step 1 within a larger function within Excel. 
5. Convert the larger function in  Excel into a new target language.

The key here is that the larger function depends on your own custom function which we are assuming the target language doesn't know anything about; however, because you have already added this to your library (either within the same file using Strict mode or manually in step 3) when spreadsheetCoder translates your larger function into the target language it also converts any parts that use the other custom function as well 

That's the gist at a high-level and a simple use-case for using the XML format to extend the functionatlity of spreadsheetCoder. Now let's dive into the details. 

## Expanding spreadsheetCoder's Functionality using XML

### Example:

Suppose you have a complex function in Excel and you translate this function to SQL for use on Teradata. Now suppose you decide to add some additional logic. You know a bit about Teradata and you happen to know that what you want would be easier with Teradata STROK function. STRTOK extracts tokens (substrings) from a string based on defined delimiters, returning the nth specified token. For the string "apple,banana,cherry", using a comma as the delimiter and requesting the 2nd token would return "banana". STRTOK is already available on Teradata but you've got your complex functionality sitting in Excel and you want to maintain it there for its easy of use, portability and easy of exaplanation in terms of how it works. 

Consider three ways to do this. All three will require you to let spreadsheetCoder know that STRTOK is a valid function in the target language (teradata) by designated the XML function as a stub function (details in the `LangSpec` section below).

### Method 1: VBA Code

Those proficient in writing VBA code, can write a VBA function mirroring the desired target language function.  In this example, develop a VBA function named `STRTOK` analogous to the Teradata version. Then, add the stub (see below) and convert your custom function to the target language. 

### Method 2: Auto-VisualBasic

If users can define the logic of the target language function in Excel, they can rely on spreadsheetCoder to generate VisualBasic code. Subsequent steps mirror Method 1.

### Method 3: Utilizing Transforms

Transforms offer an alternative. When Excel has a similar yet distinct function, a transform can bridge the gap.  `STRTOK` isn't the best example, but converting `MID` to Teradata's analogous `SUBSTRING` function is straightforward. By creating a simple transform, `MID` is converted to `SUBSTRING`.

> **Note**: 
> In 'Strict' mode, the XML function format resides in a text box on the function's page. Alternatively or in addition, storing the `.XML` format in **spreadsheetCoder/XMLFunctions** (path alterable with `XMLFunctionLibraryPath`) grants spreadsheetCoder ongoing access. This XML collection constitutes spreadsheetCoder's function library, while **spreadsheetCoder/XMLTransformations** holds transformation libraries.

# XML Functions File Documentation

## Overview

spreadsheetCoder uses XML-like files to break down intricate functions into basic, mappable elements. These definitions aid in translating embedded functions. Once stored in "spreadsheetCoder/XMLFunctions" (location adjustable via `XMLFunctionLibraryPath` in the SC file), spreadsheetCoder consistently accesses this function library.

These XML Function files are generated automatically by setting the target language to XML. The file How_to_do_lookups.xlsm is meant, as the name suggests, for help in understanding the special syntax around doing lookups with spreadsheetCoder. But it is also can be used as examples of how spreadsheetCoder generates the functional logic into other languages, including XML. You may want to refer to that file to see how things work with real examples.

Like the XML Function files, the XML Transformation files, can be generated from spreadsheetCoder itself, if you setup the right outline of the function in Excel. For examples with how this works in the file How_to_do_transforms.xlsm.

Below is the document structure for the resulting XML files.

## Key Elements:

### CodeCalculation

Root element with attributes like:
- **Name**: Defined complex function.
- **Version**: Function version.
- **HasMultipleOutputs**: Indicates if multiple outputs exist (1 for true, 0 for false).
- **AppliesTo**: Two possible values:  'All except skip' and 'Only where defined'. Not case sensitive.
  - 'All except skip' means that this XML file will be used for any language except there is a LangSpec record for that language designated Skip. 
  - 'Only where defined' means that this XML file will not be used for any language unless there is a LangSpec record for it. Even then if there is a LangSpec record for that language which says to Skip for this language it will still, of course not be used.

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

Shows the dependencies between inputs and nodes.

### FunctionNodes

Holds each functional node definition.

### ConstantNodes

Describes constants within the function.

### NamedNodes & NodeComments

These are names that are picked up from excel. It includes both the cell references as well as names it picks up from named ranges. Some language implementations can make use of these names to make it easier to correlate the functionality within the code produced to the original excel version

### Outputs

Function outputs.

### NodeDependencies

Shows the relationship between different nodes, e.g. if a node within the function represents a sum step to sum 3 other nodes and the result is used by 2 more nodes, these dependencies would capture those 5 relationships.\

# Transform File Documentation for spreadsheetCoder

## Overview

Transform files instruct spreadsheetCoder to transition one format to another. The main distinction between XML Function and Transform lies in their handling of complex scenarios.

### Key Differences: Outputs

Transform files requires defining two primary outputs in the `Outputs` section:
1. **From**: Original pattern.
2. **To**: Transformed pattern.
It can also include as many additional ouptuts as desired. each of these is considered "safe", meaning that when the transform recognizes that pattern it won't apply this transformation to any internally matching part of the transformation.

> **Note**:
> If an XML Function has a singular output, the same result can be achieved using a transform. The transformation hinges on referencing all inputs in the 'From' cell (or upstream calculations referencing the inputs) and presenting the transformed code in the 'To' cell (or upstream calculations referenceing the inputs).
