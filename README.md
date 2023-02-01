# spreadsheetCoder

Some background:
* the purpose of this excel macro excelworkbook is to allow you to turn a function you write in excel into another language.

How to:
* Prep your environment.
** You'll want a folder in your user directory titled "SpreadsheetCoder". 
** You want a subfolder of that folder named "XMLLibrary". Technically this could be empty but start with the contents in this subfolder XMLLibrary

* Now code a function from excel.
* You'll want two spreadsheets open. 1) the spreadsheet with the function in it that you want to turn into code 2) the spreadsheet coder spreadsheet.
* The spreadsheet coder excel file name needs to begin with "sc". (If you don't change the name, you're good here.)
* You will need to prep your excel function
** Put all of the input cells for your function in a range of cells arranged verically (i.e. width = 1, height = however many inputs you want)
** Label your inputs one cell to the left of each input.
** Put all of your output cells for your function in a similar range
** label your outputs for your function to the left of each output.
** give your function a name. 
*** For no particularly good reason, the name is in the following arbitrary location: 
*** put the name in the cell just above the first label for the first input cell.
** Avoid using 
*** text functions. This is really just for math stuff. 
*** indirect and functions like that
*** [placeholder for listing a lot of other limitations]
* Make sure the settings on your "Options" tab of the spreadsheet are what you want. Defaults are a good place to start.
* Run the Macro "Create". 
* It will ask you to select your input and output cells.
* When complete a popup box that says "Complete" will show. [If you are looking at the VB page you might not see it as it shows in excel.]

* That's it. In many cases the code will be saved in a file. If so, the file will be located in the SpreadsheetCoder directory referenced above.
