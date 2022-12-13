# Label_gen
Code for generating symbolic sequence labels from MuseScore files. An example is "clef-G2 + keySig-FM + timeSig-4/4 + note-A4_quarter..."

# Instructions
1. Make sure you have MuseScore installed and locate the Plugins folder for it
2. Clone and install the plugin https://github.com/ojaffe/batch_export
3. Open Plugins -> Plugin Manager in MuseScore and make sure that new plugins (Batch Convert) is enabled
4. Run the pipeline described below

## Pipeline
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

## Commands
python data/clean/removecredits.py -input data/download/files/

python data/clean/genlabels.py -input data/download/files/ -output_seq data/download/files/ -output_vocab train/vocab/ --gen_annotations

python data/clean/removetitleimgs.py -input data/download/files/

python data/clean/removenolabeldata.py -imgs data/download/files/ -labels data/download/files/

python data/clean/removesparsesamples.py -input data/download/files/

python data/clean/removewrongres.py -input data/download/files/