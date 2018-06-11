#!bin/bash

#train set list
find BBoxImage -name "*.jpg" |sort -V > bboximg.list
find JPEGImages -name "*.jpg" |sort -V> jpegimg.list
find PartImages -name "*.jpg" |sort -V >partimg.list
#test set list
find test_BBoxImage -name "*.jpg" |sort -V > test_bboximg.list
find test_PartImages -name "*.jpg" |sort -V >test_partimg.list

