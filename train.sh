project_name=SeqPred-grid-csept_256
noise_level=0.01
epochs=50
learning_rate=0.001
batch_size=64
early_stopping_patience=20
pred_length=256

for hidden_size in {8,16,32,64}; do
    for num_hidden_layers in {4,8,16,32}; do
        for num_attention_heads in {4,8,16}; do
            for num_key_value_heads in {2,4}; do
                for optimizer in {adam,sgd,adagrad}; do
                    for loss_type in {mse,mae,smooth_l1,log_cosh,huber}; do
                        python train_csept.py --hidden_size $hidden_size \
                        --num_hidden_layers $num_hidden_layers \
                        --num_attention_heads $num_attention_heads \
                        --num_key_value_heads $num_key_value_heads \
                        --train_noise_level $noise_level \
                        --pred_length $pred_length \
                        --optimizer $optimizer \
                        --epochs $epochs \
                        --batch_size $batch_size \
                        --early_stopping_patience $early_stopping_patience \
                        --learning_rate $learning_rate \
                        --project_name $project_name \
                        --loss_type $loss_type
                    done
                done
            done
        done
    done
done