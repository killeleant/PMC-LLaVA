
# Llava_Eval

生成结果的评估代码,将模型输出结果格式整理为prediction_answer.json中的格式，将所有可能的选项整理为candidate.json

## 在本地运行

Clone 这个 project

```bash
  git clone https://github.com/zyren123/Capstone.git
```

前往项目目录

```bash
  cd Capstone
```

安装依赖

```bash
  pip install -r requirements.txt 
```

启动评估(完全匹配)

```bash
python run_eval.py --gt prediction_answer.json ^
    --pred prediction_answer.json ^
    --candidate candidate.json
```
(包含匹配)
```bash
python run_eval.py --gt prediction_answer.json ^
    --pred prediction_answer.json ^
    --candidate candidate.json ^
    --strategy include
```
(使用sentence_trasnformer对所有开放式问题进行postprocess)
```bash
python run_eval.py --gt prediction_answer.json ^
    --pred prediction_answer.json ^
    --candidate candidate.json ^
    --postprocess Open
```
(使用sentence_trasnformer对所有封闭式问题进行postprocess)
```bash
python run_eval.py --gt prediction_answer.json ^
    --pred prediction_answer.json ^
    --candidate candidate.json ^
    --postprocess Close
```

(使用sentence_trasnformer对所有问题进行postprocess)
```bash
python run_eval.py --gt prediction_answer.json ^
    --pred prediction_answer.json ^
    --candidate candidate.json ^
    --postprocess Both
```

(指定使用的sentence transformer模型，按照hf中的名字为准，例如默认模型为"sentence-transformers/all-MiniLM-L6-v2",这里指定为其他模型)
```bash
python run_eval.py --gt prediction_answer.json ^
    --pred prediction_answer.json ^
    --candidate candidate.json ^
    --model sentence-transformers/all-mpnet-base-v2
```

## 也可以使用run.bat脚本，修改run.bat脚本中的参数并执行脚本

