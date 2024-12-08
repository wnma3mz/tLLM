from pathlib import Path
import time

from mflux import Config, Flux1, ModelConfig, StopImageGenerationException
from mflux.ui import defaults as ui_defaults
from mflux.ui.cli.parsers import CommandLineParser


def main():
    # fmt: off
    # parser = CommandLineParser(description="Generate an image based on a prompt.")
    # parser.add_model_arguments(require_model_arg=False)
    # parser.add_lora_arguments()
    # parser.add_image_generator_arguments(supports_metadata_config=True)
    # parser.add_image_to_image_arguments(required=False)
    # parser.add_output_arguments()
    # args = parser.parse_args()

    # Load the model
    print("Loading model...")
    s1 = time.time()
    flux = Flux1(
        model_config=ModelConfig.from_alias("schnell"),
        # quantize=4,
        # local_path="/Users/lujianghu/Documents/mflux/schnell_4bit",
        # lora_paths=args.lora_paths,
        # lora_scales=args.lora_scales,
    )
    # import mlx.core as mx
    # mx.eval(flux)
    print(f"Model loaded in {time.time() - s1:.2f}s")
    # while 1:
    #     time.sleep(1000)
    try:
        # Generate an image
        image = flux.generate_image(
            seed=42,
            prompt="a little dog",
            stepwise_output_dir=None,
            config=Config(
                num_inference_steps=2,
                height=1024,
                width=1024,
                guidance=3.5,
                init_image_path=None,
                init_image_strength=None,
            ),
        )

        # Save the image
        # image.save(path=args.output, export_json_metadata=args.metadata)
    except StopImageGenerationException as stop_exc:
        print(stop_exc)


if __name__ == "__main__":
    main()
