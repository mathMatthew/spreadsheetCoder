The purpose of this page is to give you some understanding of how spreadsheetCoder works internally.

There are multiple ways that a function in excel can get coded into the target language.
1. Direct support for the target language
The way spreadsheetCoder works, each target language has its logic (implemented as a class module in core) which takes the final steps of translating the logic of the function into the target language. As part of this process that logic can check for a specific excel function and translate it into a specific function in the target language. 

This is the way all of the languages handle the simple math functions like plus, minus, multiply etc.  Some languages also handle additional functions like this. To see an example of that look at how "ROUNDDOWN" is handled in the class module SQL_2KF.

2. Breakapart support
Before creating the specific target language code, SpreadsheetCoder breaks apart some of the function logic into more basic functional components. This breakapart support is independent of the target langauge.

For example the function sum will breakapart all of the inputs into a bunch of addition functions. 

Even though the process itself is not dependent on the target language, within the logic of a specific target language (it's class module) there is an opportunity for that class module to override the breakapart. So using the same example of sum, if a target language has a built in sum function that works like Excel's sum function (i.e. it can take any number of inputs and it will add them together), the class module can override for SUM so that it isn't broken apart as part of that process. This will then give the class module the ability to code that directly.

3. XML Library support
If a function has a fixed number of inputs, you can add support for it to SpreadsheetCoder if you can describe the logic with functions that SpreadsheetCoder already supports. I have already done this for a number of functions and these are part of the  XML library on github.

Though you have to be careful not to create circular dependencies, one of the functions within the library can include a function that is also in the library.

For example, PV is a handy function to calculate the present value of an annuity. If you look at the excel help for this function it will tell you how to use it and by following the link for more information you can see the formula Excel uses to calculate the value.

Suppose you have logic for a function in Excel and you want to translate that to run on Teradata. The function in Excel makes use of PV and for this discussion let's suppose PV wasn't already defined in the XML Library. 

Given that you know the formula for PV (you can see it in the Excel help) you could in your Excel file get rid of PV and just use the more basic calculations to calcule the present value; however, this will make your code harder to read, more error prone and will get tedious if you use PV multiple times and in different functions. So what you can do instead is you can create a function that mimics what the built-in Excel PV does. Then set your target language to XML and create the XML file that incorporates that. You'll get a file like the PV.XML file that is already in the XML Function library. But if it weren't already there, you would then put the file you created there and that means that now you can use Excel's PV function in any function you need to translate into your target language and spreadsheetCoder behind the scenes will translate it into the target language at the appropriate spots. From the perspective of the resulting code it will be the same as if you had removed the PV function from your Excel calculations and replaced them with the formula directly that Excel uses to calculate PV.
