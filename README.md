# spreadsheetCoder

Some background:
* The purpose of this Excel macro is to allow you to convert a function you've written in Excel into another programming language.

### 1. Environment Preparation:
- **Directory Structure**: Create a folder in your user directory titled "SpreadsheetCoder".
  - Move the `sc.xlsm` file into the created folder.
  - Within "SpreadsheetCoder", create subfolders: "XMLLibrary" and "XMLTransforms".
  - Copy provided contents into the respective folders.
- **Macro Settings**: 
  - Open `sc.xlsm` and [enable the macros](https://support.microsoft.com/en-us/topic/a-potentially-dangerous-macro-has-been-blocked-0952faa0-37e7-4316-b61d-5b5ed6024216).
  - In the Trust Center, enable 'Trust access to the VBA project object model'.
    - **Caution**: Beware of malware. Only run trusted macros when this option is on. More details are in the troubleshooting section.

### 2. Coding a Function in Excel:
- **Spreadsheet Selection**: Open two spreadsheets:
  1. The spreadsheet containing the function you wish to convert.
  2. The `spreadsheetCoder` file. Don't rename. The name must start with "sc".
  
- **Function Organization**: 
  - **Inputs**:
    - Arrange input cells vertically.
    - Label them to the left.
  - **Outputs**:
    - Organize output cells in the same vertical manner.
    - Label them on the left side.
    - If there's a single output, its label is the function name.
    - For multiple outputs, place the function name directly above the label of the first output.
  
- **Coding Restrictions**: 
  - Avoid text manipulation functions.
  - Refrain from using indirect or unconventional functions.
  - Do not use functions that operate on cell ranges. 
    - Exceptions include `sum`, `product`, and some lookup functions. 

### 3. Execution:
- **Settings Check**: Ensure the settings in the "Options" tab of your spreadsheet are as desired. Begin with default settings.
- **Macro Execution**: 
  - Run the "Create" Macro from your worksheet containing the target function.
  - If unfamiliar with running macros, refer to the troubleshooting section below.
  - During execution, select your input and output cells as prompted.
  - Use strict mode is preferred as it provides advantages but also requires test case presence. For test case creation, see [How_to_do_lookups.xlsm](./How_to_do_lookups.xlsm).
  
- **Completion**: A "Complete" popup will appear once the macro has finished. If you are viewing the VB window you may miss it as it shows in Excel.

- **Output**: By default spreadsheetCoder will generate a VBA function in your spreadsheet that replicates your Excel function. If you change the settings to generate VBA or another coding language as a file, the generated code will save within the "SpreadsheetCoder" directory. If you execute the macro again with the same function name, it overwrites the previous file. Rename or copy the old one to keep it.

Enjoy.

Troubleshooting
* How to run a macro.
	* Enable the Developer ribbon in Excel
	* Click on the "File" tab in the upper left corner of the Excel window.
		* In the File menu, click on "Options" at the bottom of the navigation pane. This will open the Excel Options window.
		* In the Excel Options window, click on "Customize Ribbon" on the left sidebar.
		* On the right side of the Excel Options window, you'll see a list of tabs. Check the box next to "Developer" to enable the Developer tab.
		* Click the "OK" button to save your changes and close the Excel Options window.
	* Go to the developer tab and choose Visual Basic
	* Go to the Develop ribbon and click the 'Macros' button
	* Select the macro you want (create) and then click the 'run' button
* How to enable 'trust access to the vba project model'.
	* If you're using Excel 2010 or later click on the "File" tab, then click on "Options" near the bottom of the navigation pane.
	* You'll see a list of categories on the left side. Click on "Trust Center".
	* You'll see a "Microsoft Excel Trust Center" heading with a button "Trust Center Setting" under and to the right. Click on that.
    	* You'll see a heading "Developer Macro Settings". click the checkbox "Trust access to the VBA project object model"
* References. You may want to double check that all the references are correct. Go to the Visual Basic Window, Tools->References). Validate that all of these are checked: 1) Visual Basic for Applications 2) Microsoft Excel 16.0 Object Library 3) OLE Automation 4) Microsoft Office 16.0 Object Library 5) Microsoft XML, v6.0, 6) Microsoft Visual Basic fo Applications Extensibility 5.3


--23862686275127358676078431734411442979792948070027924026
