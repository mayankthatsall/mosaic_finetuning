import argparse
import os
import torch
import transformers
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path",
                        type=str,
                        help="Path to the Composer state file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="avalara_bert/",
        help=
        "Path to the output directory that will contain `pytorch_model.bin` and `config.json`"
    )
    return parser.parse_args()

def main(args):
    """ 
    Converts a Composer checkpoint to a HF checkpoint.
    Example Usage: python convert_composer_states_to_hf.py --checkpoint-path <CHECKPOINT_PATH_HERE> --output-dir custom_trained_bert
    Then, you can load the model using the `transformers` package:
        import transformers
        transformers.AutoModelForMaskedLM.from_pretrained("custom_trained_bert/")
    and it should successfully work.
    """
    # load the Composer state
    composer_state = torch.load(args.checkpoint_path, map_location="cpu")
    # consume the `module.` prefix created by DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        composer_state['state']['model'], "module.")
    # make the HF output directory if it doesn't already exist
    os.makedirs(args.output_dir, exist_ok=True)
    # save the pytorch_model.bin file
    hf_model_path = os.path.join(args.output_dir, "pytorch_model.bin")
    torch.save(composer_state['state']['model'], hf_model_path)
    # load and save the config.json file
    #config = transformers.AutoConfig.from_pretrained("roberta-base")
    #config_path = os.path.join(args.output_dir, "config.json")
    #config.to_json_file(config_path)
    #print(f"Successfully saved {args.checkpoint_path} to {args.output_dir}.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
