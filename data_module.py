  import torch                                                                                                                                                                               
  ckpt = torch.load("checkpoints_hcodec/weights.pt", map_location="cpu")
  for k, v in ckpt.items():
      if "quantizer" in k or "embed" in k or "codebook" in k:
          print(k, v.shape)
