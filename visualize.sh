python -m visualize_attention \
  --config configs/config_self_attn_cross_attn_mogcn_data_brca.yaml \
  --checkpoint lightning_logs/brca_self_attn_cross_attn_fusion_mogcn_data_fold_4/version_0/checkpoints/epoch=19-step=260.ckpt \
  --output-dir attention_plots/grad_importance \
  --top-k-features 20 \
  --use-true-label \
  --class-names "LumA,LumB,HER2,Basal"