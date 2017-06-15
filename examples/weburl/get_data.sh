#!/bin/bash

DIR=$(dirname $0)
if [ ! -f $DIR/weburl.train ]; then
   echo "Retrieving Data..."
   wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/url_combined.bz2
   echo "Decompressing..."
   bunzip2 url_combined.bz2
   echo "Splitting into train/validation sets"
   head -n 500000 url_combined > weburl.train
   tail -n 200000 url_combined > weburl.test  
   rm url_combined
fi
     
