# 生成
uv run  batch_run.py --d-orig-dir ../srs-docs/resources/req_md_fast5/  --r-base-dir ../srs-docs/resources/summary/minimal --d-base-dir ../srs-docs/resources/baseline/docs_minimal_dsc/ --parallel 78  --output-dir output/minimal_en_iter8 --skip-existing --max-outer-iter 8
# 评估
uv run test/script/eval_batch_outputs.py   --outputs-dir output/minimal_en_iter8/  --d-orig-dir  ../srs-docs/resources/req_md   --eval-output-dir eval_reports/minimal_en_iter8/ --srs-eval-runner uv   --max-parallel 15
