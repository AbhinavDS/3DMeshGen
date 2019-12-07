#!/bin/bash
rm -rf ../unittest_data
mkdir ../unittest_data
python3 polygonGenerate.py -d 1 -f ../unittest_data -n 1 --seeded --no_draw --suffix unittest
