import os
import sys
from pathlib import Path

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up environment variables for inference
os.environ.setdefault('EVAL_DATA_PATH', 'TencentGR_1k')
os.environ.setdefault('EVAL_RESULT_PATH', 'inference_results')

# Specify a specific model checkpoint
os.environ['MODEL_OUTPUT_PATH'] = 'checkpoints/global_step70.valid_loss=1.3869'

# Ensure necessary directories exist
Path(os.environ['EVAL_RESULT_PATH']).mkdir(parents=True, exist_ok=True)

# Print environment variables for debugging
print(f"EVAL_DATA_PATH: {os.environ.get('EVAL_DATA_PATH')}")
print(f"EVAL_RESULT_PATH: {os.environ.get('EVAL_RESULT_PATH')}")
print(f"MODEL_OUTPUT_PATH: {os.environ.get('MODEL_OUTPUT_PATH')}")

# Import and run the inference function
from infer import infer

if __name__ == "__main__":
    print("Starting inference...")
    try:
        print("Starting inference...")
        top10s, user_list = infer()
        print("Inference completed.")
        print(f"Processed {len(user_list)} users.")
        if top10s:
            print(f"Generated recommendations for {len(top10s)} users.")
        else:
            print("No recommendations generated.")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        # 尝试重新运行推理以获取更多调试信息
        try:
            print("Retrying inference with more debugging info...")
            top10s, user_list = infer()
        except Exception as retry_e:
            print(f"Retry failed with error: {retry_e}")
            import traceback
            traceback.print_exc()