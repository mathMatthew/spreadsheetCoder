# Understanding How `spreadsheetCoder` Works Internally

`spreadsheetCoder` translates Excel functions into the target language code. This document aims to explain the mechanisms behind this transformation.

## 1. **Direct Support for the Target Language**
Each target language within `spreadsheetCoder` has its own logic, implemented as a class module in core. This logic facilitates the translation of specific Excel functions into their equivalents in the target language.

- **Example:** All languages translate basic math functions (e.g., plus, minus, multiply) directly. However, some might handle additional functions too. For a detailed example, observe the handling of "ROUNDDOWN" in the `SQL_2KF` class module.

## 2. **Breakapart Support**
Before generating the specific target language code, `spreadsheetCoder` dissects some function logic into simpler functional units. This process remains consistent irrespective of the target language.

- **Example:** The `SUM` function in Excel breaks apart its inputs into individual addition functions. Though the breaking apart mechanism doesn't change based on the target language, each target language's logic (its class module) can choose to override it. So, if a language supports a `SUM` function similar to Excel's, its class module can directly use that instead of allowing the breakapart process to convert it to a set of additions which get translated into the language that way.

## 3. **XML Library Support**
For functions with a fixed number of inputs, you can enhance `spreadsheetCoder` by defining the logic using functions it already recognizes. Several such functions are part of the XML library on GitHub.

**Note:** Avoid creating circular dependencies. It's permissible for one library function to reference another as long as dependencies don't loop.

- **Example:** The Excel function `PV` calculates the present value of an annuity. If `PV` wasn't pre-defined in the XML Library, and you wanted to use it as part of a function that is translated into a target language like Teradata, you could of course do the following manual method.

    1. Understand the formula behind Excel's `PV` from its help documentation.
    2. In your Excel file, you could replace `PV` with its underlying calculations.

But, this work around would complicate your code in Excel. To avoid this you could add to the XML library. (We are assumign for this example it isn't already there--though it is). 

    1. Design a function in Excel that mirrors Excel's `PV`. 
    2. In your spreadsheetCoder settings, change the target language to XML
    3. Generate the corresponding XML file. Because you named the output "PV", and there is a single output, the file will be named `PV.XML` and will be saved to your spreadsheetCoder folder.
    4. Move the XML file to the XML Function library. 

After these steps whenever you use Excel's `PV`, `spreadsheetCoder` will translate it using the underlying formula represented in the XML file in the library.
