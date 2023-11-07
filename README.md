# PFED
Profiling free emanations detection

## Python libraries to install for code to run
yaml, sys, os, pick, dumpy, pytictoc, decimal, spicy.signal, copy, math

## How to run
Copy the files manually to a directory.
Copy the required IQ files that you wish to process. Details of sample IQ files can be found below in the page. Then run the following python file.
python TopLevelFile.py
The results that include pickle files and pdf plots are saved in the Results folder created in the same directory where the python files are run from.

## IQ data location
IQ files location in [link](https://drive.google.com/drive/folders/1Hx9GdysKPMb-so-tDZ4Bj20Lt9gqTRW1?usp=sharing)

## IQ data collection setup
Laptop is connectd to Monitor. This setiup is inside a shield room. A signal hound places outside is used to collet IQ from the emanations leaking out the setup inside the shield room.
![DataCollectionSetup (1)](https://github.com/venkateshsathya/PFED/assets/54123622/cde0d2d1-932f-4b49-826c-2eaff7c74130)

## System block digram
![SystemBockDiagram (1)](https://github.com/venkateshsathya/PFED/assets/54123622/08c725e4-76be-4d04-9902-9a9d8e698efe)

## Algorithm flow chart
![EmanationDetectionFlowChart](https://github.com/venkateshsathya/PFED/assets/54123622/7a7fd028-b662-41b3-95a8-eb8073ab4200)

## Time it takes to process one 25 MHz file
![Screen Shot 2023-10-27 at 2 09 13 PM](https://github.com/venkateshsathya/PFED/assets/54123622/00dbbed7-b492-4080-9547-16e6a1442819)
On a macbook with above configurations, it takes two minutes to process each file
