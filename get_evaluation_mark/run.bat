@echo off

python run_eval.py ^
    --gt prediction_answer.json ^
    --pred prediction_answer.json ^
    --candidate candidate.json ^
    --postprocess Open
