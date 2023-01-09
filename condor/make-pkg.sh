#!/bin/bash
# NOTE: do not run from condor/, run from ./

name="afad"
# chmod +x sim/size-wq-5.0.0-1.R
tar -czvf to-transfer-$name.tar.gz datasets/$name/ models/
# tar -czvf to-transfer-$name.tar.gz datasets/$name/ models/ tfhub/ nltk_data/ # for amrev, imdb