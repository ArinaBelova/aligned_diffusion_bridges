H_LIST=(
    # "0.9"
    # "0.8"
    # "0.7"
    # "0.6"
    "0.5"
    # "0.4"
    # "0.3"
    # "0.2"
    # "0.1"
)

K_LIST=(
 "0"
# "0"
#"10"
 )

NORM_LIST=(    
'True'
)

ID_LIST=(    
'1'
#'True'
#'True'
#'True'
#'True'
# 'False'
)

g_LIST=(
"0.7"
)

CHILD_SCRIPT="train_eval.sh"

# Iterate over each set of arguments
for K in "${K_LIST[@]}"; do
        for norm in "${NORM_LIST[@]}"; do
            for H in "${H_LIST[@]}"; do
                for ID in "${ID_LIST[@]}"; do
                    for g in "${g_LIST[@]}"; do
                        echo "Launching child script with arguments: K=$K, H=$H, norm=$norm", g=$g;
                        sbatch -p gpu1 ./${CHILD_SCRIPT} $K $norm $H $ID $g;
done
done
done
done
done
