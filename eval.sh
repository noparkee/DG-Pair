echo "Model: " $1
echo "Save Paht: output/"$1

python evaluate_model.py --model-config ./config/$1.json --test-env 0 --checkpoint output/$1/testenv_3_checkpoint.pt --output-path output/$1 --trial-seed 0 &&
python evaluate_model.py --model-config ./config/$1.json --test-env 1 --checkpoint output/$1/testenv_2_checkpoint.pt --output-path output/$1 --trial-seed 0 &&
python evaluate_model.py --model-config ./config/$1.json --test-env 2 --checkpoint output/$1/testenv_1_checkpoint.pt --output-path output/$1 --trial-seed 0 &&
python evaluate_model.py --model-config ./config/$1.json --test-env 3 --checkpoint output/$1/testenv_0_checkpoint.pt --output-path output/$1 --trial-seed 0
