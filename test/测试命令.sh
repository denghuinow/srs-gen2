# 生成
uv run  batch_run.py --d-orig-dir ../srs-docs/resources/req_md  --r-base-dir ../srs-docs/resources/summary/minimal --d-base-dir ../srs-docs/resources/baseline/docs_minimal_dsc/ --parallel 78  --output-dir output/minimal_en_iter8_inner10_skip_improver --skip-existing --max-outer-iter 8
# 评估
uv run test/script/eval_batch_outputs.py   --outputs-dir output/minimal_en_iter8_inner10_skip_improver/  --d-orig-dir  ../srs-docs/resources/req_md   --eval-output-dir eval_reports/minimal_en_iter8_inner10_skip_improver/ --srs-eval-runner uv   --max-parallel 15
# 合并迭代阶段通过项，1个评委通过就算通过
uv run eval_batch_outputs.py   --outputs-dir output/minimal_en_iter8_inner10_skip_improver/  --d-orig-dir  ../srs-docs/resources/req_md   --eval-output-dir eval_reports/minimal_en_iter8_inner10_skip_improver_merge_passes_1judge_pass_Loose/ --srs-eval-runner uv --max-parallel 15 --merge-iter-passes --min-judges-pass 1