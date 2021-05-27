# rm mlp_results.csv
# for layer in 2 5 9 17 33
# do
#     for neurons in 1 2 8 16 64 128 256
#     do
#         echo "layers: ${layer}, neurons: ${neurons}"
#         python MLP.py $layer $neurons
#     done
# done

# python MLP.py 1 1


for layer in 2 5 9 17 33
do
    for neurons in 8193
    do
        echo "layers: ${layer}, neurons: ${neurons}"
        python MLP.py $layer $neurons
    done
done