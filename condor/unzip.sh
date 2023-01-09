#!/bin/bash

# Unzips the directories returned from condor
name="afad-6.0.1_sm_"
find . -name "$name*.tar.gz" -exec tar -xzf {} \;