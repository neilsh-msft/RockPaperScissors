@if "%1"=="" echo ERROR: must specify arch: x64, x86 or ARM & goto l_exit
copy /y ..\EvalCS\bin\Debug\EvalCS.exe bin\%1\Debug\AppX
copy /y external\%1\*.dll bin\%1\Debug\AppX
copy /y ..\CNTK\CNTKLibraryManaged-2.0.dll bin\%1\Debug\AppX

:l_exit