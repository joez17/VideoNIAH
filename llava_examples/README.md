# Instructions for Adding and Running Evaluation Scripts in LLaVA-NeXT

## File Placement

You need to place three code files in the correct locations within the **LLaVA-NeXT** repository:

| File | Destination Path |
|------|-----------------|
| `model_video_niah.py` | `LLaVA-NeXT/llava/eval/model_video_niah.py` |
| `evaluation_utils.py` | `LLaVA-NeXT/scripts/video/eval/evaluation_utils.py` |
| `niah_eval.sh` | `LLaVA-NeXT/scripts/video/eval/niah_eval.sh` |

## Modify `niah_eval.sh`

After placing the files, modify `niah_eval.sh` to include the correct paths for:

- Your **code directory**
- The **model checkpoint**
- The **video data folder**
- The **annotation file path**

## Run the Evaluation Script

Execute the following command to run the evaluation:

```bash
bash scripts/video/eval/niah_eval.sh
