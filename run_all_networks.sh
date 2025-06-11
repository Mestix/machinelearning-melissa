# test_all_models.sh (only for testing purposes, not for performance)
python main.py --model cnn --epochs 1 --batch-size 64 --no-save
python main.py --model dnn --epochs 1 --batch-size 64 --no-save
python main.py --model nn --epochs 1 --batch-size 64 --no-save
python main.py --model rnn_gru --epochs 1 --batch-size 64 --no-save
python main.py --model rnn_basic --epochs 1 --batch-size 64 --no-save
python main.py --model gru_attention --epochs 1 --batch-size 64 --no-save
python main.py --model rnn_attention --epochs 1 --batch-size 64 --no-save
