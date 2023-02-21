# spreadsheetCoder

Some background:
* the purpose of this excel macro excelworkbook is to allow you to turn a function you write in excel into another language.

How to:
* Prep your environment.
** You'll want a folder in your user directory titled "SpreadsheetCoder". 
** You want a subfolder of that folder named "XMLLibrary". Technically this could be empty but start with the contents in this subfolder XMLLibrary
** Open the SC excel file from here. It has macros in it and you'll have to enable the macros to work.
** Ensure that all of the following references are checked (Visual Basic Window, Tools->References): 1) Visual Basic for Applications 2) Microsoft Excel 16.0 Object Library 3) OLE Automation 4) Microsoft Office 16.0 Object Library 5) Microsoft XML, v6.0, 6) Microsoft Visual Basic fo Applications Extensibility 5.3

* Now code a function from excel.
* You'll want two spreadsheets open. 1) the spreadsheet with the function in it that you want to turn into code 2) the spreadsheet coder spreadsheet.
* The spreadsheet coder excel file name needs to begin with "sc". (If you don't change the name, you're good here.)
* You will need to prep your excel function
** Put all of the input cells for your function in a range of cells arranged verically (i.e. width = 1, height = however many inputs you want)
** Label your inputs one cell to the left of each input.
** Put all of your output cells for your function in a similar range
** label your outputs for your function to the left of each output.
** give your function a name. If your function has a single output then the name to the left of that output will be the name of the.
** if your function has multilple outputs it is a bit more tricky.
*** For no particularly good reason, if your function has more than one output, put the function name in the following somewhat location: 
*** put the name in the cell just above the first label for the first output cell.
** Avoid using 
*** text functions. This is really just for math stuff. 
*** indirect and any weird functions like that
*** functions that operate on ranges of cells. You can get away with sum, product and some of the lookup functions. 
*** [placeholder for listing a lot of other limitations]
* Make sure the settings on your "Options" tab of the spreadsheet are what you want. Defaults are a good place to start.
* Run the Macro "Create" from your worksheet that has the funciton in it. 
* It will ask you to select your input and output cells.
* When complete a popup box that says "Complete" will show. [If you are looking at the VB page you might not see it as it shows in excel.]

* That's it. Depending on your settings the code will be saved in a file. If so, the file will be located in the SpreadsheetCoder directory referenced above. If you re-run it again with the same functionName it will save over the old one, so if you want to keep the old one make sure to rename or copy it before running the create macro again for that function.

Enjoy.

--23862686275127358676078431734411442979792948070027924026
