# H_LIST=(
#     "0.5"
# )

# K_LIST=(    
# "0"
# )

# H_LIST=(
#     "0.4"
#     "0.3"
#     "0.2"
#     "0.1"
# )

# K_LIST=(    
# "1"
# "2"
# "3"
# "4"
# )

H_LIST=(
    "0.9"
    "0.8"
)

K_LIST=(    
"1"
"2"
"3"
"4"
"5"
)

CHILD_SCRIPT="train_eval.sh"

# Iterate over each set of arguments
for K in "${K_LIST[@]}"; do
        for H in "${H_LIST[@]}"; do
                echo "Launching child script with arguments: K=$K, H=$H";
                sbatch -p gpu3,gpu4 ./${CHILD_SCRIPT} $K $H;
done
done
