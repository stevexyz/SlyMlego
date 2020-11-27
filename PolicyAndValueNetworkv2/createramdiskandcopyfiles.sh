#!/bin/bash

sudo modprobe brd rd_nr=1 rd_size=20000000
ls -lah /dev/ram*
sudo zpool create tempramdisk0 ram0
sudo mkdir /tempramdisk0/PolicyAndValueNetworkv2
sudo chown steve:steve /tempramdisk0/PolicyAndValueNetworkv2/
cp -r __inputs* /tempramdisk0/PolicyAndValueNetworkv2/
cp -r __validationdata /tempramdisk0/PolicyAndValueNetworkv2/
cp TrainModel.py /tempramdisk0/PolicyAndValueNetworkv2/
cp FeaturesExtraction.py /tempramdisk0/PolicyAndValueNetworkv2/
cp Const.py /tempramdisk0/PolicyAndValueNetworkv2/
cp clean-model.sh /tempramdisk0/PolicyAndValueNetworkv2/
cp extract-model.sh /tempramdisk0/PolicyAndValueNetworkv2/
cp mv-backtoprocess.py /tempramdisk0/PolicyAndValueNetworkv2/

