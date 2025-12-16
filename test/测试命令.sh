# 生成
uv run  batch_run.py --d-orig-dir ../srs-docs/resources/req_md/  --r-base-dir ../srs-docs/resources/summary/minimal --d-base-dir ../srs-docs/resources/baseline/docs_minimal_dsc/ --parallel 78  --output-dir output/minimal_en_iter10 --skip-existing --max-outer-iter 10
# 评估
uv run test/script/eval_batch_outputs.py   --outputs-dir output/minimal_en_iter10/  --d-orig-dir  ../srs-docs/resources/req_md   --eval-output-dir eval_reports/minimal_en_iter10/ --srs-eval-runner uv   --max-parallel 15
