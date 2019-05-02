
sudo apt-get install python3-tk python-pydot python-pydot-ng graphviz
sudo -H pip3 install pydot graphviz h5py pickledb python-chess numpy matplotlib

mkdir __inputstobeprocessed
mkdir __validationdata
./PrepareInput.py sts.epd 1400
./mv-tovalidation.py

