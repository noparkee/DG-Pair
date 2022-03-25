echo "Model: " $1
echo "Save Path: output/"$2

python train_model.py --model-config ./config/$1.json --test-env 0 --output-path output/$2 --trial-seed 0 &&
python train_model.py --model-config ./config/$1.json --test-env 1 --output-path output/$2 --trial-seed 0 &&
python train_model.py --model-config ./config/$1.json --test-env 2 --output-path output/$2 --trial-seed 0 &&
python train_model.py --model-config ./config/$1.json --test-env 3 --output-path output/$2 --trial-seed 0
