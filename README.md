Testé sous Config Debug x86 ok 

Etape nécessaire 
-----------------

Installation ANN - installation des librairies dépendantes : GSL 
 Suivre le tutoriel suivant : Installation avec vcpkg   https://solarianprogrammer.com/2020/01/26/getting-started-gsl-gnu-scientific-library-windows-macos-linux/#gsl_installation_windows
 
 Si vous l'avez deja dans le dossier d'install de vcpkg : (mise à jour)  
 > git pull 
 > .\bootstrap-vcpkg.bat 
 > .\vcpkg.exe install gsl gsl:x64-windows