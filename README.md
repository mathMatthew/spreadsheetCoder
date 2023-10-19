# spreadsheetCoder

Some background:
* The purpose of this Excel macro is to allow you to convert a function you've written in Excel into another programming language.

How to:
* Prep your environment.
  * You'll want a folder in your user directory titled "SpreadsheetCoder". Move the sc.xlsm file into that folder. [You could override this and use a different folder name but let's keep it simple to get started.]
  * Create a subfolder within the "SpreadsheetCoder" folder and name it "XMLLibrary". While this folder can technically be empty, it's recommended to begin with the contents provided in the "XMLLibrary" subfolder.
  * You also need a subfolder named "XMLTransforms". Copy the files in this subfolder's XMLTransforms into that.
  * Open sc.xlsm.
    * Note: The file uses macros. Make sure to enable the macros for this to work (See [how to enable macros](https://support.microsoft.com/en-us/topic/a-potentially-dangerous-macro-has-been-blocked-0952faa0-37e7-4316-b61d-5b5ed6024216)).
  * In the Trust Center, check 'Trust access to the VBA project object model'.
    * **Note**: This module allows the creation of VBA code (and other languages). To allow this VBA code to generate more VBA code, this option must be enabled.
    * **Caution**: This setting might be exploited by malware. Do not run untrusted macros when this option is enabled. For more details on accessing the Trust Center, refer to the troubleshooting section.
* Now code a function from excel.
* You'll need to have two spreadsheets open: 1) the spreadsheet containing the function you wish to convert into code, and 2) the spreadsheet coder file.
* The spreadsheet coder excel file name needs to begin with "sc". (If you don't change the name, you're good here.)
* You will need to prep your excel function
  * Put all of the input cells for your function in a range of cells arranged verically (i.e. width = 1, height = however many inputs you want)
  * Label your inputs one cell to the left of each input.
  * Put all of your output cells for your function in a similar range
  * Label your outputs for your function to the left of each output.
  * Give your function a name. If your function has a single output then the name to the left of that output will be the name of the function.
  * If your function has multiple outputs it is a bit more tricky.
    * If your function has more than one output, put the function name in the following arbitrary location: 
    * Put the name in the cell just above the first label for the first output cell.
  * Avoid using 
    * Text manipulation functions. This is really just for math stuff. That doesn't mean you can't make it work, but it might be more trouble than its worth.
    * Indirect or any weird functions like that.
    * Functions that operate on ranges of cells. You can get away with sum, product and some of the lookup functions. 
      * [placeholder for listing a lot of other limitations]
* Make sure the settings on your "Options" tab of the spreadsheet are what you want. Defaults are a good place to start.
* Run the Macro "Create" from your worksheet that has the function in it. If you aren't sure how to run a macro see troubleshooting section below.
* It will prompt you to select your input and output cells. The default setting utilizes strict mode, which mandates the presence of test cases.
    * To create test cases, allocate one row for each test case. For each function input, design one column. For a practical example, refer to the file [How_to_do_lookups.xlsm](#).
* When complete, a popup box that says "Complete" will show. [If you are looking at the VB page you might not see it as it shows in excel.]

* That's it. Depending on your settings the code will be saved in a file. If so, the file will be located in the SpreadsheetCoder directory referenced above. If you re-run it again with the same functionName it will save over the old one, so if you want to keep the old one make sure to rename or copy it before running the create macro again for that function.

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
