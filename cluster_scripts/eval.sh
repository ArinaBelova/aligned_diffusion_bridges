#H_LIST=(
#    "0.5"
#)

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
    # "0.9"
      "0.5"
    #  "0.7"
    #  '0.6'
    #  "0.5"
    #  "0.4"
    #  "0.3"
    #  "0.2"
    #  "0.1"
)

K_LIST=(
# "0"
# "0"
# "0"
# "0"
"0"
#"6"
# "5"
# "6"
# "7"
# "8"
#"9"
#"10"
#"8"
 )
 
#H_LIST=(
    #  "0.9"
    #  "0.8"
    #  "0.7"
    #  "0.6"
    #  "0.4"
    #  "0.3"
    #  "0.2"
    #  "0.1"
#)

# K_LIST=(    
# "1"
#  "2"
#  "3"
#  "4"
#  "5"
#)
NORM_LIST=(    
'True'
#'True'
#'True'
#'True'
#'True'
# 'False'
)

CHILD_SCRIPT="train_eval.sh"

# Iterate over each set of arguments
for K in 5
        for H in "${H_LIST[@]}"; do
            for norm in "${NORM_LIST[@]}"; do
                    echo "Launching child script with arguments: K=$K, H=$H, norm=$norm";
                    sbatch -p gpu3,gpu4 ./${CHILD_SCRIPT} $K $norm $H $ID;
done
done
done
