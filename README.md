# Persona-dialogue
This is the code repository of [The Evaluation of Chinese Human-Computer Dialogue Technology (SMP2019-ECDT)](http://conference.cipsc.org.cn/smp2019/evaluation.html) Task2: Personalized dialogue. The official evaluation scripts and submission guide are released on the [codalab](https://worksheets.codalab.org/worksheets/0x8f68b61a8b2249d7b314c6e800e2dace). We achieved 3rd rank on the evaluation [leaderboard](https://adamszq.github.io/smp2019ecdt_task2/). Our code is based on Seq2Seq and pretty simple, but with high scalability.

## Requirements
- PyTorch >= 1.0
- NLTK
- tqdm
- tensorboardX

## Quickstart
### Preprocessing
Due to data license, all train/valid/test data should be accessed via the email ```smp2019ecdt@163.com```. For data preprocessing, run:
```
sh process_data.sh
```

### Training & Testing
For model training, run:
```
sh run_train.sh
```
For model testing, run:
```
sh run_test.sh
```

### Evaluation
For evalution, run:
```
sh eval_bleu.sh
sh eval_distinct.sh
sh eval_ppl.sh
```