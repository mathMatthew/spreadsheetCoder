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

- **Example with Excel's `PV` function:** 

The `PV` function in Excel calculates the present value of an annuity. Suppose it wasn't already part of the XML Library (although in reality it is), and you needed to use it in a function that translates to a target language like Teradata. 

A straightforward method would be to:

  1. Look at Excel's help documentation to find the formula behind `PV`.
  2. Directly replace the `PV` function in your Excel file with its underlying formula.

However, this approach can clutter your Excel file, making it harder to manage.

**A better alternative:** Enhance the XML library.

  1. Create a function in Excel that imitates the behavior of Excel's `PV`.
  2. Adjust your `spreadsheetCoder` settings, setting the target language to XML.
  3. Run the create macro to generate the corresponding XML file. Given the output name is "PV" and there's only one output, the file will be named `PV.XML` and stored in your `spreadsheetCoder` directory.
  4. Relocate this XML file to the XML Function library.

After these actions, each time you use Excel's `PV` function, `spreadsheetCoder` will utilize the formula encapsulated in the XML library file for translation.
