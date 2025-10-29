wandb_entity=terrencechen

wandb_project=SeqPred-grid-5dimInput
noise_level=0.01
epochs=50
learning_rate=0.001
batch_size=32
attention_dropout=0.1

for hidden_size in {8,16}; do
    for num_hidden_layers in {4,8}; do
        for num_attention_heads in {2,4}; do
            for num_key_value_heads in {2,4}; do
                for optimizer in {adam,sgd,adagrad}; do
                    for pred_length in {256,1024}; do
                        python train_csept.py --hidden_size $hidden_size \
                        --num_hidden_layers $num_hidden_layers \
                        --num_attention_heads $num_attention_heads \
                        --num_key_value_heads $num_key_value_heads \
                        --train_noise_level $noise_level \
                        --pred_length $pred_length \
                        --optimizer $optimizer \
                        --epochs $epochs \
                        --batch_size $batch_size \
                        --attention_dropout $attention_dropout \
                        --learning_rate $learning_rate \
                        --wandb_entity $wandb_entity \
                        --wandb_project $wandb_project
                    done
                done
            done
        done
    done
done