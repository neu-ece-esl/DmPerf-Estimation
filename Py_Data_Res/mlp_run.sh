for train_size in 50 100 200 500 1000 2000 5000
do
    echo "Trianing size: ${train_size}"
    python MLP.py 5 2048 $train_size 
done
