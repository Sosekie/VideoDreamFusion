- HunyuanVideoThreestudio_CUDA126
H100, CUDA126, FlashAttention-3 beta failed, FlashAttention-2 succeed
可以运行使用FlashAttention-2的推理，50步耗时13分钟，distillation版本耗时一分半

- ThreestudioWithHunyuanVideo
从现有的HunyuanVideoThreestudio_CUDA126克隆而来，配置threestudio所需库
完美运行

- HunyuanVideoThreestudio_CUDA126_H100_normalFlashAttention
H100, CUDA126, normalFlashAttention（失败了）
可以运行没有flashattention的推理，耗时30分钟

- HunyuanVideoThreestudio_CUDA126_4090
4090, CUDA126, 还没安装

