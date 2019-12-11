#!/bin/bash
rm -rf ../unittest_data
mkdir ../unittest_data
python3 polygonGenerate.py -d 2 -f ../unittest_data -n 3 --seeded --no_draw --suffix unittest --no_overlap
