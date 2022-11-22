#!/bin/bash

# Unzips the directories returned from condor
name="fgnet-1.0.2_sm_"
find . -name "$name*.tar.gz" -exec tar -xzf {} \;