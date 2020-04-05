#!/bin/bash


# Generate csv-files like the other csv-files in this directory from benchmark-file

# Requirements: python (for calculate e. g. seconds to milliseconds)

# Examples of Usage
# source generateCsv
#
# toCsv "ExampleVectorAdd" "yacxGEMMPinnend"
# toCsv "OpenCLBenchmarkCLBlastKeplerSGemmTotal" "executorKernelTime"
# toCsv "OpenCLBenchmarkKeplerSGemmExe" "executor"



#Datasizes (die vom ausprinteten Ergebnis sind ja etwas gerundet)
GEMMBenchmark="0.061527252, 0.0625, 0.5595741, 0.5625, 1.557621, 1.5625, 3.0556679, 3.0625, 5.0537148, 5.0625, 7.5517616, 7.5625, 10.5498085, 10.5625, 14.047855, 14.0625, 18.045902, 18.0625, 22.54395, 22.5625, 27.541996, 27.5625, 33.040043, 33.0625, 39.03809, 39.0625, 45.536137, 45.5625, 52.534184, 52.5625, 60.03223, 60.0625, 68.03027, 68.0625, 76.52832, 76.5625, 85.52637, 85.5625, 95.024414, 95.0625, 105.02246, 105.0625, 115.52051, 115.5625, 126.518555, 126.5625, 138.0166, 138.0625, 150.01465, 150.0625, 162.5127, 162.5625, 175.51074, 175.5625, 189.00879, 189.0625, 203.00684, 203.0625, 217.50488, 217.5625, 232.50293, 232.5625, 248.00098, 248.0625, 263.99902, 264.0625, 280.49707, 280.5625, 297.49512, 297.5625, 314.99316, 315.0625, 332.9912, 333.0625, 351.48926, 351.5625, 370.4873, 370.5625, 389.98535, 390.0625, 409.9834, 410.0625, 430.48145, 430.5625, 451.4795, 451.5625, 472.97754, 473.0625, 494.9756, 495.0625, 517.47363, 517.5625, 540.4717, 540.5625, 563.9697, 564.0625, 587.9678, 588.0625, 612.4658, 612.5625, 637.46387, 637.5625, 662.9619, 663.0625, 688.95996, 689.0625, 715.458, 715.5625, 742.45605, 742.5625, 769.9541, 770.0625, 797.95215, 798.0625, 826.4502, 826.5625, 855.44824, 855.5625, 884.9463, 885.0625, 914.94434, 915.0625, 945.4424, 945.5625, 976.4404, 976.5625, 1007.9385, 1008.0625, 1039.9365, 1040.0625, 1072.4346, 1072.5625, 1105.4326, 1105.5625, 1138.9307, 1139.0625, 1172.9287, 1173.0625, 1207.4268, 1207.5625"
CLBlast="0.0625, 0.5625, 1.5625, 3.0625, 5.0625, 7.5625, 10.5625, 14.0625, 18.0625, 22.5625, 27.5625, 33.0625, 39.0625, 45.5625, 52.5625, 60.0625, 68.0625, 76.5625, 85.5625, 95.0625, 105.0625, 115.5625, 126.5625, 138.0625, 150.0625, 162.5625, 175.5625, 189.0625, 203.0625, 217.5625, 232.5625, 248.0625, 264.0625, 280.5625, 297.5625, 315.0625, 333.0625, 351.5625, 370.5625, 390.0625, 410.0625, 430.5625, 451.5625, 473.0625, 495.0625, 517.5625, 540.5625, 564.0625, 588.0625, 612.5625, 637.5625, 663.0625, 689.0625, 715.5625, 742.5625, 770.0625, 798.0625, 826.5625, 855.5625, 885.0625, 915.0625, 945.5625, 976.5625, 1008.0625, 1040.0625, 1072.5625, 1105.5625, 1139.0625, 1173.0625, 1207.5625"
KeplerBest="0.0625, 39.0625, 150.0625, 333.0625, 588.0625, 915.0625"
KeplerS="0.0625, 0.5625, 1.5625, 3.0625, 5.0625, 7.5625, 10.5625, 14.0625, 18.0625, 22.5625, 27.5625, 33.0625, 39.0625, 45.5625, 52.5625, 60.0625, 68.0625, 76.5625, 85.5625, 95.0625, 105.0625, 115.5625, 126.5625, 138.0625, 150.0625, 162.5625, 175.5625, 189.0625, 203.0625, 217.5625, 232.5625, 248.0625, 264.0625, 280.5625, 297.5625, 315.0625, 333.0625, 351.5625, 370.5625, 390.0625, 410.0625, 430.5625, 451.5625, 473.0625, 495.0625, 517.5625, 540.5625, 564.0625, 588.0625, 612.5625, 637.5625, 663.0625, 689.0625, 715.5625, 742.5625, 770.0625, 798.0625, 826.5625, 855.5625, 885.0625, 915.0625, 945.5625, 976.5625, 1008.0625, 1040.0625, 1072.5625, 1105.5625, 1139.0625, 1173.0625, 1207.5625"
ReduceBenchmark="0.001953125, 18.001953, 36.001953, 54.001953, 72.00195, 90.00195, 108.00195, 126.00195, 144.00195, 162.00195, 180.00195, 198.00195, 216.00195, 234.00195, 252.00195, 270.00195, 288.00195, 306.00195, 324.00195, 342.00195, 360.00195, 378.00195, 396.00195, 414.00195, 432.00195, 450.00195, 468.00195, 486.00195, 504.00195, 522.00195, 540.00195, 558.00195, 576.00195, 594.00195, 612.00195, 630.00195, 648.00195, 666.00195, 684.00195, 702.00195, 720.00195, 738.00195, 756.00195, 774.00195, 792.00195, 810.00195, 828.00195, 846.00195, 864.00195, 882.00195, 900.00195, 918.00195, 936.00195, 954.00195, 972.00195, 990.00195, 1008.00195, 1026.002, 1044.002, 1062.002"
ReduceOpenCL="0.001953125, 18.001953, 36.001953, 54.001953, 72.00195, 90.00195, 108.00195, 126.00195, 144.00195, 162.00195, 180.00195, 198.00195, 216.00195, 234.00195, 252.00195, 270.00195, 288.00195, 306.00195, 324.00195, 342.00195, 360.00195, 378.00195, 396.00195, 414.00195, 432.00195, 450.00195, 468.00195, 486.00195, 504.00195, 522.00195, 540.00195, 558.00195, 576.00195"
VectorAdd="0.0625, 0.5625, 1.5625, 3.0625, 5.0625, 7.5625, 10.5625, 14.0625, 18.0625, 22.5625, 27.5625, 33.0625, 39.0625, 45.5625, 52.5625, 60.0625, 68.0625, 76.5625, 85.5625, 95.0625, 105.0625, 115.5625, 126.5625, 138.0625, 150.0625, 162.5625, 175.5625, 189.0625, 203.0625, 217.5625, 232.5625, 248.0625, 264.0625, 280.5625, 297.5625, 315.0625, 333.0625, 351.5625, 370.5625, 390.0625, 410.0625, 430.5625, 451.5625, 473.0625, 495.0625, 517.5625, 540.5625, 564.0625, 588.0625, 612.5625, 637.5625, 663.0625, 689.0625, 715.5625, 742.5625, 770.0625, 798.0625, 826.5625, 855.5625, 885.0625, 915.0625, 945.5625, 976.5625, 1008.0625, 1040.0625, 1072.5625, 1105.5625, 1139.0625, 1173.0625, 1207.5625"

getDataSize(){
    dataSizeNumber=$1
    
    data=""
    if [ "$(echo $exampleName | grep 'GEMMBenchmark')" != "" ]; then
        data="$GEMMBenchmark"
    fi
    if [ "$(echo $exampleName | grep 'CLBlast')" != "" ]; then
        data="$CLBlast"
    fi
    if [ "$(echo $exampleName | grep 'KeplerBest')" != "" ]; then
        data="$KeplerBest"
    fi
    if [ "$(echo $exampleName | grep 'KeplerS')" != "" ]; then
        data="$KeplerS"
    fi
    if [ "$(echo $exampleName | grep 'ReduceBenchmark')" != "" ]; then
        data="$ReduceBenchmark"
    fi
    if [ "$(echo $exampleName | grep 'ReduceOpenCL')" != "" ]; then
        data="$ReduceOpenCL"
    fi
    if [ "$(echo $exampleName | grep 'VectorAdd')" != "" ]; then
        data="$VectorAdd"
    fi
    
    IFS=" "
    set "$data"

    echo "$(echo "$data" | cut -d " " -f $dataSizeNumber | sed 's/,//')"
}

calculateUnits(){
    [ "$exeTimeUnit" == "s" ] && exeTime=$(python -c "print($exeTime*1000)")
    [ "$exeTimeUnit" == "m" ] && exeTime=$(python -c "print($exeTime*1000*60)")

    [ "$totTimeUnit" == "s" ] && totTime=$(python -c "print($totTime*1000)")
    [ "$totTimeUnit" == "m" ] && totTime=$(python -c "print($totTime*1000*60)")

    [ "$upTimeUnit" == "s" ] && upTime=$(python -c "print($upTime*1000)")
    [ "$upTimeUnit" == "m" ] && upTime=$(python -c "print($upTime*1000*60)")

    [ "$dwTimeUnit" == "s" ] && dwTime=$(python -c "print($dwTime*1000)")
    [ "$dwTimeUnit" == "m" ] && dwTime=$(python -c "print($dwTime*1000*60)")
}

splitLineYacx1(){
    IFS=" "
    set $line

    exeTime=$4
    exeTimeUnit=$5
    totTime=$8
    totTimeUnit=$9
    upTime=${11}
    upTimeUnit=${12}
    dwTime=${14}
    dwTimeUnit=${15}
}

splitLineYacx2(){
    IFS=" "
    set $line

    exeTime=$6
    exeTimeUnit=$7
    totTime=${10}
    totTimeUnit=${11}
    upTime=${13}
    upTimeUnit=${14}
    dwTime=${16}
    dwTimeUnit=${17}
}

splitLineOpenCL(){
    IFS=" "
    set $line

    exeTime=$4
    exeTimeUnit=$5
    totTime=""
    totTimeUnit=""
    upTime=""
    upTimeUnit=""
    dwTime=""
    dwTimeUnit=""
}

toCsv(){
    exampleName=$1
    branchName=$2

    inputFile="benchmarkTestTitan.dat"
    outputFile=$(echo ${exampleName}_${branchName}.csv)

    echo "%$1 $2" > $outputFile
    if [ "$branchName" != "executor" ]; then
        echo "Datasize execution-time total-time upload-time download-time" >> $outputFile
    else
        echo "Datasize execution-time" >> $outputFile
    fi

    echo "" >> $outputFile

    lineStart=$(grep -hn ": $exampleName in $branchName$" $inputFile | cut -d: -f1)
    if [ "$branchName" != "executor" ]; then
        lines=$(tail -n +$lineStart $inputFile | grep -hn "Benchmark-Duration:" | head -n 1| cut -d: -f1)
    else
        lines=$(tail -n +$lineStart $inputFile | grep -hn "Benchmark exeuted in" | head -n 1| cut -d: -f1)
    fi

    lineStart=$((lineStart + 5))
    lines=$((lines - 7))

    if [ "$(echo $exampleName | grep 'Open')" != "" ] && [ "$branchName" != "executor" ]; then
        lineStart=$((lineStart + 1))
        lines=$((lines - 1))
    fi

    i="1"
    while IFS= read -r line; do
        if [ "$branchName" != "executor" ]; then
            if [ "$(echo $line | grep 'matrices')" == "" ]; then
                splitLineYacx1 "$line"
            else
                splitLineYacx2 "$line"
            fi
        else
            splitLineOpenCL
        fi

        calculateUnits
        
        echo "$(getDataSize $i) $exeTime $totTime $upTime $dwTime" >> $outputFile
        
        i=$((i + 1))
    done <<< "$(tail -n +$lineStart $inputFile | head -n $lines | sed -e 's/^[ \t]*//' | sed 's/,//' | sed 's/,//' | sed 's/)//')"
}
