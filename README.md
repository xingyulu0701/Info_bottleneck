# Code based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

## Missing Environment Dependencies

Environment code is taking from a co-author's ongoing project, and is temporarily removed.

## Requirements

* Python 3 
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt
```

## Training Scripts
### Cartpole:
Deterministic Encoder:
```
python main.py --env-name OracleTrain --algo ppo --state --deterministic --save-dir cp --num-env-steps 1000000
```
Low noise:
```
python main.py --env-name OracleTrain --algo ppo --state --beta 0.000000001 --load-dir cp/ppo/OracleTrain.sh --save-dir cp_dkl --num-env-steps 1000000
```
Anneal:
```
python main.py --env-name OracleTrain --algo ppo --state --beta 0.000000001 --beta-end 1 --load-dir cp_dkl/ppo/OracleTrain.sh --save-dir cp_dkl_anneal --num-env-steps 10000000
```


### HalfCheetah:

Deterministic Encoder:
```
python main.py --env-name HCTrain --algo ppo --state --deterministic --save-dir hc --num-env-steps 3000000 --code-size 128
```
Low noise:
```
python main.py --env-name HCTrain --algo ppo --state --beta 0.000000001 --load-dir hc/ppo/HCTrain.sh --save-dir hc_dkl --num-env-steps 3000000
```
Anneal:
```
python main.py --env-name HCTrain --algo ppo --state --beta 0.000000001 --beta-end 1 --load-dir hc_dkl/ppo/HCTrain.sh --save-dir hc_dkl_anneal --num-env-steps 20000000
```

### Humanoid:
Deterministic Encoder:
```
python main.py --env-name SHTrain --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 5e-6 --entropy-coef 0 --value-loss-coef 1 --ppo-epoch 10 --num-mini-batch 64 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 3000000 --use-proper-time-limits --state --deterministic --save-dir sh --code-size 96 
```

Anneal:
```
python main.py --env-name SHTrain  --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 5e-6 --entropy-coef 0 --value-loss-coef 1 --ppo-epoch 10 --num-mini-batch 64 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 10000000 --use-proper-time-limits --state --beta 0.000000001 --beta-end 1 --load-dir sh/ppo/SHTrain.pt --save-dir sh_anneal --code-size 96 
```

## Evaluation:
```
python main.py --env-name {ENV} --algo ppo --state --lr 0 --load-dir {model} --code-size {code-size} --eval-interval 1

```

