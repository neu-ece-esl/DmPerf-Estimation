# rm mlp_results.csv
for layer in 2 5 9 17 33
do
    for neurons in 1 2 8 16 64 128 256 512 1024 2048 4096 8192
    do
        echo "layers: ${layer}, neurons: ${neurons}"
        python MLP.py $layer $neurons 1000
    done
done

python MLP.py 1 1
