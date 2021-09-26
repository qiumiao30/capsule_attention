

#optimize: adam, nada, adadelta, SGD
#loss: bce, focal_bce, mccfocal, dicebce, combo
#data_name: vsb
#model:cnn_lstm, lstm_attention, capsule_attention

#data_name:vsb
###optimizer
python main.py --model cnn_lstm --batch_size 64 --seq_len 64 --data_name vsb --loss bce --optimizer adam
python main.py --model cnn_lstm --batch_size 64 --seq_len 64 --data_name vsb --loss bce --optimizer nada
python main.py --model cnn_lstm --batch_size 64 --seq_len 64 --data_name vsb --loss bce --optimizer adadelta
python main.py --model cnn_lstm --batch_size 64 --seq_len 64 --data_name vsb --loss bce --optimizer SGD

python main.py --model lstm_attention --batch_size 64 --seq_len 64 --data_name vsb --loss bce --optimizer adam
python main.py --model lstm_attention --batch_size 64 --seq_len 64 --data_name vsb --loss bce --optimizer nada
python main.py --model lstm_attention --batch_size 64 --seq_len 64 --data_name vsb --loss bce --optimizer adadelta
python main.py --model lstm_attention --batch_size 64 --seq_len 64 --data_name vsb --loss bce --optimizer SGD


###loss--cnn_lstm
python main.py --model cnn_lstm --batch_size 64 --seq_len 64 --data_name vsb --loss bce --optimizer adam
python main.py --model cnn_lstm --batch_size 64 --seq_len 64 --data_name vsb --loss focal_bce --optimizer adam
python main.py --model cnn_lstm --batch_size 64 --seq_len 64 --data_name vsb --loss mccfocal --optimizer adam
python main.py --model cnn_lstm --batch_size 64 --seq_len 64 --data_name vsb --loss dicebce --optimizer adam
python main.py --model cnn_lstm --batch_size 64 --seq_len 64 --data_name vsb --loss combo --optimizer adam

###loss--lstm_attention
python main.py --model lstm_attention --batch_size 64 --seq_len 64 --data_name vsb --loss bce --optimizer adam
python main.py --model lstm_attention --batch_size 64 --seq_len 64 --data_name vsb --loss focal_bce --optimizer adam
python main.py --model lstm_attention --batch_size 64 --seq_len 64 --data_name vsb --loss mccfocal --optimizer adam
python main.py --model lstm_attention --batch_size 64 --seq_len 64 --data_name vsb --loss dicebce --optimizer adam
python main.py --model lstm_attention --batch_size 64 --seq_len 64 --data_name vsb --loss combo --optimizer adam

###loss--capsule_attention
python main.py --model capsule_attention --batch_size 64 --seq_len 64 --data_name vsb --loss bce --optimizer adam
python main.py --model capsule_attention --batch_size 64 --seq_len 64 --data_name vsb --loss focal_bce --optimizer adam
python main.py --model capsule_attention --batch_size 64 --seq_len 64 --data_name vsb --loss mccfocal --optimizer adam
python main.py --model capsule_attention --batch_size 64 --seq_len 64 --data_name vsb --loss dicebce --optimizer adam
python main.py --model capsule_attention --batch_size 64 --seq_len 64 --data_name vsb --loss combo --optimizer adam

