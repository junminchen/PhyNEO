#!/bin/bash
iframe=$1
for i in $iframe_*out; do
    grep SETTING\ EINT $i | cut -c 27-45
done
