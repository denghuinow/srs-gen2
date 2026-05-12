# 8次迭代
# 生成
cd /root/project/srs/srs-gen3 && uv run  batch_run.py --d-orig-dir ../srs-docs/resources/req_md  --r-base-dir ../srs-docs/resources/summary/ultra_short --d-base-dir ../srs-docs/resources/baseline/docs_ultra_short_dsc/ --parallel 20  --output-dir output/20ultra_short_iter8_skip_improver --skip-existing --max-outer-iter 8   --files \
  "2003 - agentmom.pdf" \
  "2005 - triangle.pdf" \
  "2001 - elsfork.pdf" \
  "2007 - central trading system.pdf" \
  "2005 - clarus low.pdf" \
  "2009 - model manager.pdf" \
  "2010 - gparted.pdf" \
  "0000 - inventory.pdf" \
  "2009 - video search.pdf" \
  "2004 - philips.doc" \
  "2010 - home 1.3.pdf" \
  "2003 - qheadache.pdf" \
  "2008 - vub.pdf" \
  "2005 - nenios.html" \
  "2009 - email.pdf" \
  "2009 - gaia.pdf" \
  "1998 - themas.pdf" \
  "2007 - mdot.pdf" \
  "2001 - libra.doc" \
  "2007 - e-store.doc"
# 评估
# 合并迭代阶段通过项，1个评委通过就算通过
cd /root/project/srs/srs-gen3 && uv run eval_batch_outputs.py   --outputs-dir output/20ultra_short_iter8_skip_improver/  --d-orig-dir  ../srs-docs/resources/req_md   --eval-output-dir eval_reports/20ultra_short_iter8_skip_improver_merge_passes_1judge_pass_Loose/ --srs-eval-runner uv --max-parallel 20 --merge-iter-passes --min-judges-pass 1

cd /root/project/srs/srs-gen3 && python3 statistics_iter.py --eval-reports-dir eval_reports/20ultra_short_iter8_skip_improver_merge_passes_1judge_pass_Loose --output-dir output/20ultra_short_iter8_skip_improver
