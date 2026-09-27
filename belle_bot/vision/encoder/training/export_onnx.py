import argparse
import torch
import torch.onnx
from belle_bot.vision.encoder.training.ml_model import VAE2_448
import os


class EncoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        mu, _ = self.vae.encode(x)
        return mu


class DecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z)
    

def export_onnx(model: VAE2_448, output_path):
    # get the current device and set it back at the end
    original_device = next(model.parameters()).device
    
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 4, 448, 448, device=device)

    target_model = EncoderWrapper(model)
    input_names = ["input"]
    output_names = ["embedding"]

    encoder_path = output_path.replace(".onnx", "_encoder.onnx") if output_path.endswith(".onnx") else output_path + "_encoder.onnx"
    torch.onnx.export(
        target_model,
        dummy_input,
        encoder_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={input_names[0]: {0: 'batch_size'},
                      output_names[0]: {0: 'batch_size'}}
    )
    
    target_model = DecoderWrapper(model)
    dummy_input = torch.randn(1, model.latent_dim, device=device)
    input_names = ["embedding"]
    output_names = ["output"]
    
    decoder_path = output_path.replace(".onnx", "_decoder.onnx") if output_path.endswith(".onnx") else output_path + "_decoder.onnx"
    torch.onnx.export(
        target_model,
        dummy_input,
        decoder_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={input_names[0]: {0: 'batch_size'},
                      output_names[0]: {0: 'batch_size'}}
    )

    # Restore the original device
    model.to(original_device)

    # Verification and Consolidation
    try:
        import onnx
        for path in [encoder_path, decoder_path]:
            # Load the model to check it and also to ensure it's consolidated into one file
            onnx_model = onnx.load(path)
            onnx.checker.check_model(onnx_model)
            
            # Explicitly save it to a single file to avoid .onnx.data files
            onnx.save_model(onnx_model, path, save_as_external_data=False)
            
            # Clean up any .data file if it was created by the initial export
            data_path = path + ".data"
            if os.path.exists(data_path):
                os.remove(data_path)
                print(f"Removed external data file: {data_path}")

    except Exception as e:
        print(f"ONNX verification/consolidation failed: {e}")
