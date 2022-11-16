#!/bin/bash
# NOTE: do not run from condor/, run from ./

name="fgnet"
# chmod +x sim/size-wq-5.0.0-1.R
tar -czvf to-transfer-$name.tar.gz datasets/$name/ models/
