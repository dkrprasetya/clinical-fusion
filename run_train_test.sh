python main.py --task los_bin > lstm-ts-los_bin-train.out
python main.py --task los_bin --phase test --resume > lstm-ts-los_bin-test.out
cp models/lstm.model models/lstm-ts-los_bin.model

python main.py --task readmit > lstm-ts-readmit-train.out
python main.py --task readmit --phase test --resume > lstm-ts-readmit-test.out
cp models/lstm.model models/lstm-ts-readmit.model


python main.py --model cnn --task mortality > cnn-ts-mortality-train.out
python main.py --model cnn --task mortality --phase test --resume > cnn-ts-mortality-test.out
cp models/cnn.model models/cnn-ts-mortality.model

python main.py --model cnn --task los_bin > cnn-ts-los_bin-train.out
python main.py --model cnn --task los_bin --phase test --resume > cnn-ts-los_bin-test.out
cp models/cnn.model models/cnn-ts-los_bin.model

python main.py --model cnn --task readmit > cnn-ts-readmit-train.out
python main.py --model cnn --task readmit --phase test --resume > cnn-ts-readmit-test.out
cp models/cnn.model models/cnn-ts-readmit.model