#!/bin/bash

echo "=== Testing CUDA Auto-Detection ==="

echo "1. Testing normal detection:"
make clean 2>&1 | grep -E "(Detected|Selected)"

echo ""
echo "2. Testing manual override:"
CUDA_PATH=/usr/local/cuda-13.0 make clean 2>&1 | grep -E "(Using manually|Detected|Selected)"

echo ""
echo "3. Available CUDA installations:"
ls -d /usr/local/cuda* 2>/dev/null

echo ""
echo "4. Current driver version:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1

echo ""
echo "5. Driver-supported CUDA version:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | \
    awk 'BEGIN{FS="."} { 
        if ($1 >= 580) print "13.0"; 
        else if ($1 >= 570) print "12.8"; 
        else if ($1 >= 560) print "12.6"; 
        else if ($1 >= 550) print "12.4"; 
        else if ($1 >= 535) print "12.2"; 
        else if ($1 >= 525) print "12.0"; 
        else if ($1 >= 515) print "11.8"; 
        else if ($1 >= 510) print "11.6"; 
        else if ($1 >= 495) print "11.4"; 
        else if ($1 >= 470) print "11.2"; 
        else if ($1 >= 460) print "11.0"; 
        else print "10.2"; 
    }'

echo ""
echo "=== Test Complete ==="