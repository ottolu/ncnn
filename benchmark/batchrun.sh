
list=`cat $1`
latency=$(awk 'BEGIN{print 0.0 }')

IFS=$'\n'
for line in $list
do 
    IFS=$' '
    cmd="./testkernel $line"
    echo $cmd
    file=`$cmd`
    l=${file:0:8}
    #l=$file
    echo $l
    latency=$(awk "BEGIN{print $latency+$l }")
done

echo $latency


