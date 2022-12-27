# Polyphonic-Omr
 PyTorch code for end-to-end Optical Music Recognition (OMR) on polyphonic scores. Based off [the following repository](https://github.com/sachindae/polyphonic-omr), which was a result from the paper:

 "An Empirical Evaluation of End-to-End Polyphonic Optical Music Recognition"
 
 ## Released Items
 - [x] Dataset Creation
 - [ ] Model Training
 - [ ] Model Inference
 - [ ] Downloadable Weights

## Dataset Creation
If you don't have a dataset already, we support the creation of it by scraping and processing MuseScore files, found [here](https://github.com/Xmader/musescore-dataset). This will convert scraped .musicxml files into an image sequence of symbolic labels, an example of which is "clef-G2 + keySig-FM + timeSig-4/4 + note-A4_quarter...

Requirements:
1. Make sure you have [MuseScore](https://musescore.org/en) installed and locate the Plugins folder for it
2. Clone and install the plugin https://github.com/ojaffe/batch_export
3. Open Plugins -> Plugin Manager in MuseScore and make sure that new plugins (Batch Convert) is enabled

For downloading the data we support Linux and Windows. Run the correct script in ./data/download/

Then preprocess the files in ./data/clean/ with the following commands:
1. Run "Batch Convert" in MuseScore on all .mscz files to .musicxml, select "Resize Export"
2. Run removecredits.py on the generated .musicxml files
3. Run "Batch Convert" in MuseScore on the cleaned .musicxml files to .mscz
4. Run "Batch Convert" in MuseScore on the new .mscz to .musicxml and .png
5. Run genlabels.py to generate labels for the .musicxml files
6. Run the following to clean the data, as needed:
    - removetitleimgs.py
    - removenolabeldata.py
    - removenonpolyphonic.py 
    - removesparsesamples.py

## Training
TODO
