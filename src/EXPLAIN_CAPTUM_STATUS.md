# Explain Captum Status

## Working Methods ✅

| Method                 | Status   | Notes                                                  |
| ---------------------- | -------- | ------------------------------------------------------ |
| `saliency`             | ✅ Works | Gradient-based attribution                             |
| `integrated_gradients` | ✅ Works | Path-integrated gradients with convergence delta       |
| `layer_gradcam`        | ✅ Works | Returns mostly zeros - may need different target layer |

## Not Working ❌

| Method             | Error                                                            | Fix Required                              |
| ------------------ | ---------------------------------------------------------------- | ----------------------------------------- |
| `deep_lift`        | `'function' object has no attribute 'register_forward_pre_hook'` | Wrap forward function in proper nn.Module |
| `guided_backprop`  | `Given model must be an instance of torch.nn.Module`             | Use model directly instead of wrapper     |
| `guided_gradcam`   | Same as guided_backprop                                          | Use model directly                        |
| `input_x_gradient` | `One of the differentiated Tensors does not require grad`        | Needs proper embedding gradient setup     |
| `noise_tunnel`     | Timeout                                                          | Depends on slow methods                   |

## Too Slow ⏳

| Method             | Issue                                              |
| ------------------ | -------------------------------------------------- |
| `feature_ablation` | Embedding dimension (768) makes ablation very slow |

## GPU Issues

- Some methods fail on GPU with CUDA index errors
- CPU works reliably for saliency, integrated_gradients, layer_gradcam

## Next Steps

1. Implement a custom `nn.Module` wrapper that exposes the model properly for Captum hooks
2. Try different layer for LayerGradCam (currently using last encoder layer)
3. Test on faster GPU (A100/H100) for feature ablation

## Usage

```bash
# Run working methods
python src/explain_captum.py --methods saliency integrated_gradients layer_gradcam

# Use specific text
python src/explain_captum.py --methods saliency --text "tu eres muy bonita"

# Save results
python src/explain_captum.py --methods saliency --output results.json
```
