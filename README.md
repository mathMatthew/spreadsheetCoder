# spreadsheetCoder

Some background:
* the purpose of this excel macro excelworkbook is to allow you to turn a function you write in excel into another language.

How to:
* Prep your environment.
  * You'll want a folder in your user directory titled "SpreadsheetCoder". Move the sc.xlsm file into that folder.
  * You want a subfolder of that folder named "XMLLibrary". Technically this could be empty but start with the contents in this subfolder's XMLLibrary
  * Open sc.xlsm. Note the file uses macros and you'll have to [enable the macros for this to work.](https://support.microsoft.com/en-us/topic/a-potentially-dangerous-macro-has-been-blocked-0952faa0-37e7-4316-b61d-5b5ed6024216)

* Now code a function from excel.
* You'll want two spreadsheets open. 1) the spreadsheet with the function in it that you want to turn into code 2) the spreadsheet coder spreadsheet.
* The spreadsheet coder excel file name needs to begin with "sc". (If you don't change the name, you're good here.)
* You will need to prep your excel function
  * Put all of the input cells for your function in a range of cells arranged verically (i.e. width = 1, height = however many inputs you want)
  * Label your inputs one cell to the left of each input.
  * Put all of your output cells for your function in a similar range
  * Label your outputs for your function to the left of each output.
  * Give your function a name. If your function has a single output then the name to the left of that output will be the name of the.
  * If your function has multiple outputs it is a bit more tricky.
    * For no particularly good reason, if your function has more than one output, put the function name in the following somewhat location: 
    * Put the name in the cell just above the first label for the first output cell.
  * Avoid using 
    * text functions. This is really just for math stuff. 
    * indirect and any weird functions like that
    * functions that operate on ranges of cells. You can get away with sum, product and some of the lookup functions. 
      * [placeholder for listing a lot of other limitations]
* Make sure the settings on your "Options" tab of the spreadsheet are what you want. Defaults are a good place to start.
* Run the Macro "Create" from your worksheet that has the funciton in it. If you aren't sure how to run a macro see troubleshooting section below.
* It will ask you to select your input and output cells.
* When complete a popup box that says "Complete" will show. [If you are looking at the VB page you might not see it as it shows in excel.]

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
Go to the developer tab and choose Visual Basic
 * Go to the Develop ribbon and click the 'Macros' button
 * Select the macro you want (create) and then click the 'run' button
* References. You may want to double check that all the references are correct. Go to the Visual Basic Window, Tools->References). Validate that all of these are checked: 1) Visual Basic for Applications 2) Microsoft Excel 16.0 Object Library 3) OLE Automation 4) Microsoft Office 16.0 Object Library 5) Microsoft XML, v6.0, 6) Microsoft Visual Basic fo Applications Extensibility 5.3


--23862686275127358676078431734411442979792948070027924026
